import torch as t
from torch import Tensor
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
from tqdm import tqdm
from functools import partial
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Tuple
import torch as t
import re
from collections import defaultdict


import torch as t
import torch.nn.functional as F

@t.inference_mode()
def avg_logprob_for_label(model, tokenizer, prompt: str, label: str, device) -> float:
    """
    Average log-probability of `label` given `prompt`, using teacher forcing.
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    label_ids  = tokenizer.encode(label,  add_special_tokens=False)
    if len(label_ids) == 0:
        return 0.0

    # Run on prompt+label; logits[t] predicts token at t+1
    input_ids = t.tensor([prompt_ids + label_ids], device=device)
    logits = model(input_ids).logits  # [1, T_total, V]

    pl = len(prompt_ids)
    L  = len(label_ids)

    # Positions that predict the label tokens
    # first label token predicted at index pl-1, then pl,...,pl+L-2
    to_score = logits[0, pl-1 : pl-1+L, :]              # [L, V]
    log_probs = F.log_softmax(to_score, dim=-1)         # [L, V]

    label_ids_t = t.tensor(label_ids, device=device)    # [L]
    tok_logp = log_probs.gather(1, label_ids_t[:, None]).squeeze(1)  # [L]

    return float(tok_logp.mean().item())


from torch.nn.utils.rnn import pad_sequence
@t.inference_mode()
def batch_avg_logprob_for_labels(model, tokenizer, prompts, labels, device, batch_size=16):
    """
    Returns a scalar float: the mean (over examples) of the average
    log-probabilities of each label given its prompt (teacher forcing).
    """
    assert len(prompts) == len(labels)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    out_vals = []

    for s in range(0, len(prompts), batch_size):
        P  = prompts[s:s+batch_size]
        Ls = labels[s:s+batch_size]

        # tokenize
        prom_ids = [t.tensor(tokenizer.encode(p,  add_special_tokens=False), dtype=t.long) for p in P]
        labl_ids = [t.tensor(tokenizer.encode(lb, add_special_tokens=False), dtype=t.long) for lb in Ls]

        # concat prompt+label per example
        concat = [t.cat([pi, li], dim=0) for pi, li in zip(prom_ids, labl_ids)]
        prompt_lens = t.tensor([len(pi) for pi in prom_ids], device=device, dtype=t.long)
        label_lens  = t.tensor([len(li) for li in labl_ids], device=device, dtype=t.long)

        # pad to batch
        input_ids = pad_sequence(concat, batch_first=True, padding_value=pad_id).to(device)  # [B, T_max]
        outputs = model(input_ids)
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, (tuple, list)):
            logits = outputs[0]
        else:
            logits = outputs
        log_probs = F.log_softmax(logits, dim=-1)  # [B, T_max, V]

        B, T_max, _ = log_probs.shape
        L_max = max(1, int(label_lens.max().item()))

        # pad label ids to L_max to index cleanly
        label_pad = pad_sequence(labl_ids, batch_first=True, padding_value=pad_id).to(device)  # [B, L_max]

        # positions to read for each label token: start = pl-1, then +0..L_i-1
        arange_L = t.arange(L_max, device=device)[None, :].expand(B, -1)   # [B, L_max]
        starts   = (prompt_lens - 1).unsqueeze(1)                           # [B, 1]
        pos_idx  = (starts + arange_L).clamp(min=0, max=T_max-1)            # [B, L_max]

        # mask only real label tokens (ignore padded positions)
        mask = arange_L < label_lens.unsqueeze(1)                           # [B, L_max]

        # gather log-probs of the correct next tokens
        b_idx    = t.arange(B, device=device).unsqueeze(1).expand_as(pos_idx)  # [B, L_max]
        tok_logp = log_probs[b_idx, pos_idx, label_pad]                        # [B, L_max]

        # zero out pads, then average over real label length
        tok_logp    = tok_logp * mask
        denom       = label_lens.clamp_min(1).to(tok_logp.dtype)               # [B]
        per_ex_mean = tok_logp.sum(dim=1) / denom                               # [B]

        out_vals.append(per_ex_mean.detach().cpu())

    all_vals = t.cat(out_vals, dim=0)  # [N]
    return float(all_vals.mean().item())

# -----------------------------------------------------------
# 1) Core: compute sequence log-prob for the *correct* label
#    (optionally masking pad tokens)
# -----------------------------------------------------------
def compute_seq_logprob(
    logits: t.Tensor,
    answer_token_seqs: t.Tensor,
    *,
    reduction: str = "mean",     # "mean" over tokens; could be "sum" if you prefer
    array: bool = False,         # return per-item values if True, else batch mean
    pad_token_id: int = None,
):
    """
    Compute (masked) average log-probability of the *correct* label sequence.

    Args:
        logits: [batch, seq_len, vocab_size]
        answer_token_seqs:
            Either [batch, 2, label_len] (correct, incorrect) as produced by
            `get_answer_token_sequences`, or [batch, label_len] (only correct).
        reduction: "mean" (default) or "sum" over the label tokens.
        array: if True, return a 1D tensor [batch] of per-item values; else return batch mean (scalar).
        pad_token_id: if provided, positions equal to this id are masked out.

    Returns:
        Scalar (batch mean) or a 1D tensor [batch] if array=True.
    """
    assert reduction in {"mean", "sum"}

    # Accept both shapes: [B, 2, L] or [B, L]
    if answer_token_seqs.dim() == 3:
        # Use only the correct sequence channel
        correct_labels = answer_token_seqs[:, 0, :]          # [B, L]
    else:
        correct_labels = answer_token_seqs                    # [B, L]

    B, L = correct_labels.shape
    # Slice the last L positions from logits (labels are aligned to the end)
    logits_to_score = logits[:, -L:, :]                      # [B, L, V]
    log_probs = F.log_softmax(logits_to_score, dim=-1)       # [B, L, V]

    gathered = log_probs.gather(2, correct_labels.unsqueeze(-1)).squeeze(-1)  # [B, L]

    if pad_token_id is not None:
        mask = (correct_labels != pad_token_id)              # [B, L]
        # avoid division by zero; if a row is fully padded, we define its mean as 0
        lengths = mask.sum(dim=1).clamp_min(1)               # [B]
        gathered = gathered * mask                           # zero-out pads
        if reduction == "mean":
            per_item = gathered.sum(dim=1) / lengths         # [B]
        else:
            per_item = gathered.sum(dim=1)                   # [B]
    else:
        if reduction == "mean":
            per_item = gathered.mean(dim=1)                  # [B]
        else:
            per_item = gathered.sum(dim=1)                   # [B]

    if array:
        return per_item
    return per_item.mean()


# -----------------------------------------------------------
# 2) Batched helper (used to be get_batched_logit_diff)
# -----------------------------------------------------------
def get_batched_logprob(minibatch_size, tokens, answer_tokens, model, pad_token_id=None):
    """
    Returns the average sequence log-prob (correct label only) over the dataset.
    """
    avg = 0.0
    n = 0
    for i in range(0, len(tokens), minibatch_size):
        logits, _ = model.run_with_cache(tokens[i:i+minibatch_size])
        at = answer_tokens[i:i+minibatch_size]
        val = compute_seq_logprob(logits, at, pad_token_id=pad_token_id)  # scalar
        # accumulate numerically stable
        batch_size = at.size(0)
        avg = (avg * n + float(val) * batch_size) / (n + batch_size)
        n += batch_size
        del logits, val
    return avg


## Intervention layer wise / focusing on layer wise position of input tokens

def patching_residual_hook(corrupted_residual_component, hook, pos, clean_cache):
    corrupted_residual_component[:, pos, :] = clean_cache[hook.name][:, pos, :]
    return corrupted_residual_component

@t.inference_mode()
def patching_residual(model, corrupted_tokens, clean_cache, patching_metric, answer_tokens, clean_logit_diff, corrupted_logit_diff, device):
    model.reset_hooks()
    seq_len = corrupted_tokens.size(1)
    results = t.zeros(model.cfg.n_layers, seq_len, device=device, dtype=t.float32)

    for layer in tqdm(range(model.cfg.n_layers)):
        for position in range(seq_len):
            hook_fn = partial(patching_residual_hook, pos=position, clean_cache=clean_cache)
            patched_logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks = [(utils.get_act_name("resid_post", layer), hook_fn)],
            )
            results[layer, position] = patching_metric(patched_logits, answer_tokens, corrupted_logit_diff, clean_logit_diff)
    return results

## Intervention head wise / focusing on each head in the model
def patching_attention_hook(corrupted_head_vector, hook, head_index, clean_cache):
    corrupted_head_vector[:, :, head_index] = clean_cache[hook.name][:, :, head_index]
    return corrupted_head_vector

@t.inference_mode()
def patching_attention(model, corrupted_tokens, clean_cache, patching_metric, answer_tokens, clean_logit_diff, corrupted_logit_diff, head_type, device):
    model.reset_hooks()
    results = t.zeros(model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=t.float32)

    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            hook_fn = partial(patching_attention_hook, head_index=head, clean_cache=clean_cache)
            patched_logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks = [(utils.get_act_name(head_type, layer), hook_fn)],
                return_type="logits"
            )
            results[layer, head] = patching_metric(patched_logits, answer_tokens, corrupted_logit_diff, clean_logit_diff)

    return results

## Ablation
def accumulated_zero_ablation(clean_head_vector, hook, head_list):
    heads_to_patch = [head for layer, head in head_list if layer == hook.layer()]
    if len(heads_to_patch) == 0:
        return clean_head_vector
    clean_head_vector[:, :, heads_to_patch, :] = 0
    return clean_head_vector

def accumulated_mean_ablation(clean_head_vector, hook, head_list):
    heads_to_patch = [head for layer, head in head_list if layer == hook.layer()]
    if len(heads_to_patch) == 0:
        return clean_head_vector
    clean_head_vector[:, :, heads_to_patch, :] = clean_head_vector.mean(dim=0, keepdim=True)[:, :, heads_to_patch, :]
    return clean_head_vector

def get_accumulated_ablation_score(model, labels, tokens, answer_tokens, head_list, clean_logit_diff, type, device):
    model.reset_hooks()
    results = t.zeros(model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=t.float32)

    if head_list == None:
      return clean_logit_diff, None, None

    if type == 'mean':
        hook_fn = accumulated_mean_ablation
    else:
        hook_fn = accumulated_zero_ablation

    head_type = 'z'
    head_layers = set(next(zip(*head_list)))
    hook_names = [utils.get_act_name(head_type, layer) for layer in head_layers]
    hook_names_filter = lambda name: name in hook_names

    hook_fn = partial(
        hook_fn,
        head_list=head_list
    )
    patched_logits = model.run_with_hooks(
        tokens,
        fwd_hooks = [(hook_names_filter, hook_fn)],
        return_type="logits"
    )

    return compute_logit_diff(patched_logits, answer_tokens)

def necessity_check(model, labels, tokens, answer_tokens, clean_logit_diff, type, device):
    sequence = [(23,10), (19,1), (18, 12), (17,2), (15, 14), (14,14), (11, 10), (7,2), (6, 15), (6,1), (5,8)]
    target_head = []
    scores = []

    for head in tqdm(sequence):
        target_head.append(head)
        score = get_accumulated_ablation_score(
            model, labels, tokens, answer_tokens, target_head, clean_logit_diff, type, device
        )
        scores.append(score)

    scores = [clean_logit_diff] + scores
    for i in range(len(scores)):
        scores[i] = scores[i].cpu()
        scores[i] = scores[i].cpu()
    return scores

def sufficiency_check(model, labels, tokens, answer_tokens, clean_logit_diff, type, device):
    sequence = [(23,10), (19,1), (18, 12), (17,2), (15, 14), (14,14), (11, 10), (7,2), (6, 15), (6,1), (5,8)]
    all_heads = [(i, j) for i in range(24) for j in range(16)]
    all_score = get_accumulated_ablation_score(
        model, labels, tokens, answer_tokens, all_heads, clean_logit_diff, type, device
    )
    target_head = []
    scores = []

    for head in tqdm(sequence[::-1]):
        all_heads_temp = all_heads.copy()
        target_head.append(head)
        for th in target_head:
            if th in all_heads_temp:
                all_heads_temp.remove(th)
        score = get_accumulated_ablation_score(
            model, labels, tokens, answer_tokens, all_heads_temp, clean_logit_diff, type, device
        )
        scores.append(score)

    scores = [all_score] + scores
    for i in range(len(scores)):
        scores[i] = scores[i].cpu()
    return scores


def get_label_token_lengths(dataset):
    """
    Compute the maximum token length for each symbolic label across the dataset,
    accounting for variable tokenization of a, b, and c (and verbs).

    Returns:
        dict: { label: max_len }
              Labels: ["BEGIN","s_1","s_1 -> m_1","m_1","m_1 -> m_2",
                       "m_2","m_2 -> p","p","p -> s_2","s_2","END"]
    """
    tokenizer = dataset.tokenizer
    labels = [
        "BEGIN",
        "s_1",
        "s_1 -> m_1",
        "m_1",
        "m_1 -> m_2",
        "m_2",
        "m_2 -> p",
        "p",
        "p -> s_2",
        "s_2",
        "END"
    ]

    max_len = {lab: 0 for lab in labels}

    # Regex to extract a, v1, b, v2, c, v3 based on your prompt builder
    # Example prompt:
    # "Since A VERB1 B and B VERB2 C, therefore A VERB3"

    # Helper to get token length
    def tlen(text): 
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])
    
    max_len["BEGIN"] = tlen("Since")
    max_len["m_1 -> m_2"] = tlen(" and ")
    max_len["p -> s_2"] = tlen(", therefore")
    max_len["END"] = tlen(" => ")


    prompt_lens = []

    for prompt in dataset.prompts:

        prompt_len = {lab: 0 for lab in labels}
        prompt_len["BEGIN"] = tlen("Since ")
        prompt_len["m_1 -> m_2"] = tlen(" and ")
        prompt_len["p -> s_2"] = tlen(", therefore ")



        # s_1: a
        prompt_len["s_1"] = tlen(prompt["a"]+ " ")
        max_len["s_1"] = max(max_len["s_1"],prompt_len["s_1"])

        # s_1 -> m_1: v1
        prompt_len["s_1 -> m_1"] =tlen(prompt["v1"]+ " ")
        max_len["s_1 -> m_1"] = max(max_len["s_1 -> m_1"], prompt_len["s_1 -> m_1"])

        # m_1: b
        prompt_len["m_1"] =  tlen(prompt["b"])
        max_len["m_1"] = max(max_len["m_1"], prompt_len["m_1"])

        # m_2: b2
        prompt_len["m_2"] =  tlen(prompt["b2"]+ " ")
        max_len["m_2"] = max(max_len["m_2"], prompt_len["m_2"])

        # m_2 -> p: ", therefore" (comma + space + word)
        prompt_len["m_2 -> p"] = tlen(prompt["v2"]+ " ")
        max_len["m_2 -> p"] = max(max_len["m_2 -> p"], prompt_len["m_2 -> p"])

        # p: a
        prompt_len["p"] = tlen(prompt["label"])
        max_len["p"] = max(max_len["p"], prompt_len["p"])

        

        # s_2: c  (your earlier code maps s_2 to c again)
        prompt_len["s_2"] = tlen(prompt["a"])
        max_len["s_2"] = max(max_len["s_2"], prompt_len["s_2"])

        # END: no EOS added with add_special_tokens=False; keep 0 or set to 1 if you want a column
        prompt_len["END"] = tlen(" => ")
        prompt_lens.append(prompt_len)

    return prompt_lens, max_len

def build_label_spans(max_len_by_label: Dict[str, int], labels: List[str]) -> Dict[str, Tuple[int, int]]:
    """
    Given max token lengths per label and an ordered label list, return
    cumulative [start, end) spans for each label along the sequence axis.
    """
    spans = {}
    cur = 0
    for lab in labels:
        L = int(max_len_by_label.get(lab, 0))
        spans[lab] = (cur, cur + L)
        cur += L
    return spans


def compress_by_label_spans(
    tensor: t.Tensor,
    spans: Dict[str, Tuple[int, int]],
    labels: List[str],
    mode: str = "mean",
) -> t.Tensor:
    """
    Aggregate a sequence-by-layer tensor into label rows using spans.

    Accepts `tensor` with shape:
        - [seq_len, n_layers]  OR
        - [n_layers, seq_len]  (will be transposed internally)

    Returns:
        compressed: [len(labels), n_layers] (CPU tensor)
    """
    assert mode in {"mean", "sum"}
    x = tensor
    if x.dim() != 2:
        raise ValueError("Expected a 2D tensor (seq_len x n_layers or n_layers x seq_len).")

    

    seq_len, n_layers = x.shape
    out = []

    for lab in labels:
        s, e = spans[lab]
        s_clamped = max(0, min(s, seq_len))
        e_clamped = max(0, min(e, seq_len))

        if e_clamped <= s_clamped:
            # No tokens allocated for this label → zeros row
            out.append(t.zeros(n_layers, dtype=x.dtype, device=x.device))
        else:
            seg = x[s_clamped:e_clamped, :]  # [seg_len, n_layers]
            agg = seg.mean(dim=0) if mode == "mean" else seg.sum(dim=0)
            out.append(agg)

    compressed = t.stack(out, dim=0)   # [11, n_layers]
    return compressed.detach().cpu()  


import torch as t

def make_pad_keymask(tokens: t.Tensor, pad_token_id: int):
    if not isinstance(pad_token_id, int):
        raise TypeError(f"pad_token_id must be int, got {type(pad_token_id)}")
    if tokens.ndim == 1:  # allow [T]
        tokens = tokens.unsqueeze(0)
    is_pad = (tokens == pad_token_id)                 # [B, T], bool
    return is_pad[:, None, None, :]                   # [B, 1, 1, T]

def make_attn_mask_hook(key_pad_mask: t.Tensor):
    def hook(scores, hook):
        mask = key_pad_mask.to(scores.device)         # scores: [B,H,Q,K]
        return scores.masked_fill(mask, 0.0)  # mask pad **keys**
    return hook

def run_ignoring_pad(model, tokens: t.Tensor, pad_token_id: int):
    key_pad_mask = make_pad_keymask(tokens, pad_token_id)  # [B,1,1,T]
    hook_fn = make_attn_mask_hook(key_pad_mask)
    fwd_hooks = [(f"blocks.{l}.attn.hook_attn_scores", hook_fn)
                 for l in range(model.cfg.n_layers)]
    # Apply hooks here, then run_with_cache
    with model.hooks(fwd_hooks=fwd_hooks):
        logits, cache = model.run_with_cache(tokens)
    return logits, cache


def get_adjusted_token_sequences(dataset, max_len) -> t.Tensor:


    tokenizer = dataset.tokenizer
    tokenised_sentences = []

    begin_tokens = tokenizer("Since ", add_special_tokens=False)["input_ids"]
    first_transition_tokens = tokenizer(" and ", add_special_tokens=False)["input_ids"]
    second_transition_tokens = tokenizer(", therefore ", add_special_tokens=False)["input_ids"]
    #end_tokens = tokenizer(" => ", add_special_tokens=False)["input_ids"]

    for prompt in dataset.prompts:
        tokenised_sentence = []
        tokenised_sentence += begin_tokens
        tokenised_sentence += tokenizer(prompt["a"], add_special_tokens=False,padding="max_length",max_length=max_len["s_1"], truncation=True)["input_ids"]
        tokenised_sentence += tokenizer(prompt["v1"], add_special_tokens=False,padding="max_length",max_length=max_len["s_1 -> m_1"], truncation=True)["input_ids"]
        tokenised_sentence += tokenizer(prompt["b"], add_special_tokens=False,padding="max_length",max_length=max_len["m_1"], truncation=True)["input_ids"]
        tokenised_sentence += first_transition_tokens
        tokenised_sentence += tokenizer(prompt["b2"], add_special_tokens=False,padding="max_length",max_length=max_len["m_2"], truncation=True)["input_ids"]
        tokenised_sentence += tokenizer(prompt["v2"], add_special_tokens=False,padding="max_length",max_length=max_len["m_2 -> p"], truncation=True)["input_ids"]
        tokenised_sentence += tokenizer(prompt["label"], add_special_tokens=False,padding="max_length",max_length=max_len["p"], truncation=True)["input_ids"]
        tokenised_sentence += second_transition_tokens
        tokenised_sentence += tokenizer(prompt["a"], add_special_tokens=False,padding="max_length",max_length=max_len["s_1"], truncation=True)["input_ids"]
        tokenised_sentence += tokenizer(prompt["v3"], add_special_tokens=False,padding="max_length",max_length=max_len["s_1"], truncation=True)["input_ids"]


        tokenised_sentences.append(tokenised_sentence)  # don't stack yet

    tokens = t.tensor(tokenised_sentences, dtype=t.long)  # [B, T_fixed]
    return tokens



def get_answer_token_sequences(labels: list[str], second_labels: list[str], tokenizer, max_len=None) -> t.Tensor:
    """
    Returns a tensor of shape [batch_size, 2, label_len] containing tokenized
    correct and incorrect labels, padded to the same length.

    Args:
        labels: List of correct label strings.
        second_labels: List of incorrect label strings.
        tokenizer: HuggingFace tokenizer.
        max_len: Optional fixed label length to pad/truncate to.

    Returns:
        Tensor of shape [batch_size, 2, label_len]
    """
    assert len(labels) == len(second_labels), "Mismatched label lengths"

    token_pairs = []

    for correct, incorrect in zip(labels, second_labels):
        correct_ids = tokenizer(correct, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        incorrect_ids = tokenizer(incorrect, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        token_pairs.append([correct_ids, incorrect_ids])  # don't stack yet

    # Flatten, pad, reshape
    flat = [x for pair in token_pairs for x in pair]  # [batch * 2]
    padded = pad_sequence(flat, batch_first=True, padding_value=tokenizer.pad_token_id)  # [batch*2, max_len]
    padded = padded.view(len(labels), 2, -1)  # [batch, 2, max_len]

    if max_len is not None:
        cur_len = padded.size(-1)
        if cur_len > max_len:
            padded = padded[:, :, :max_len]
        elif cur_len < max_len:
            pad_width = max_len - cur_len
            pad_tensor = t.full((padded.size(0), 2, pad_width), tokenizer.pad_token_id, dtype=padded.dtype)
            padded = t.cat([padded, pad_tensor], dim=-1)

    return padded


## Misc
def normalise_tensor(tensor):
    max_abs_val = t.max(t.abs(tensor))
    normalised_tensor = tensor / max_abs_val
    return normalised_tensor

def resize_all(tensor):
    def resize_tensor(tensor, pre, post):
        tensor[:, pre] += tensor[:, post]
        resized_tensor = np.delete(tensor, pre, axis=1)
        return resized_tensor
    temp = resize_tensor(tensor[:].cpu(), 11, 12)
    temp = resize_tensor(temp, 10, 11)
    temp = resize_tensor(temp, 9, 10)
    temp = resize_tensor(temp, 4, 5)
    return temp

def compute_logit_diff(logits: t.Tensor, answer_token_seqs: t.Tensor, array=False):
    """
    Computes normalized logit difference for multi-token answer sequences.

    Args:
        logits: Tensor of shape [batch, seq_len, vocab_size] — from model forward pass.
        answer_token_seqs: Tensor of shape [batch, 2, label_len] — correct and incorrect label token IDs.
        array: If True, return individual logit differences per sample; else, return the mean.

    Returns:
        Float (mean logit diff) or array of diffs, depending on `array`.
    """
    # Get relevant logits at the end of the sequence (i.e., the label tokens)
    label_len = answer_token_seqs.size(-1)
    logits_to_score = logits[:, -label_len:, :]  # [batch, label_len, vocab_size]
    log_probs = F.log_softmax(logits_to_score, dim=-1)  # [batch, label_len, vocab_size]

    # Extract correct and incorrect label token sequences
    correct_labels = answer_token_seqs[:, 0, :]  # [batch, label_len]
    incorrect_labels = answer_token_seqs[:, 1, :]  # [batch, label_len]

    # Gather log-probs for the correct and incorrect tokens
    correct_log_probs = log_probs.gather(2, correct_labels.unsqueeze(-1)).squeeze(-1)  # [batch, label_len]
    incorrect_log_probs = log_probs.gather(2, incorrect_labels.unsqueeze(-1)).squeeze(-1)  # [batch, label_len]

    # Normalize: take mean log-prob over the label length
    correct_mean = correct_log_probs.mean(dim=1)  # [batch]
    incorrect_mean = incorrect_log_probs.mean(dim=1)  # [batch]

    logit_diff = correct_mean - incorrect_mean  # [batch]

    if array:
        return logit_diff.cpu().numpy()

    return logit_diff.mean()

def get_batched_logit_diff(minibatch_size, tokens, answer_tokens, model):
    avg_logit_diff = 0
    for i in range(0, len(tokens), minibatch_size):
        target_index = i
        logit, _ = model.run_with_cache(tokens[target_index: target_index+minibatch_size])
        at = answer_tokens[target_index: target_index+minibatch_size]
        logit_diff = compute_logit_diff(logit, at)
        avg_logit_diff += logit_diff
        del logit
        del logit_diff
    return avg_logit_diff/(len(tokens)/minibatch_size)

def align_fine_tuning_gpt2(model, f_model, device):
    model.W_E.copy_(f_model.transformer.wte.weight)
    are_close = t.allclose(model.W_E, f_model.transformer.wte.weight)
    print(f"Are the tensors close?(token embedding) {are_close}")

    model.W_pos.copy_(f_model.transformer.wpe.weight)
    are_close = t.allclose(model.W_pos, f_model.transformer.wpe.weight)
    print(f"Are the tensors close?(position embedding) {are_close}")

    # number of layers
    for i in range(0, model.cfg.n_layers):

        # store the temp value
        # attention weight
        temp_Q = f_model.transformer.h[i].attn.c_attn.weight[..., :1024].reshape(1024, 16, 64).permute(1, 0, 2)
        temp_K = f_model.transformer.h[i].attn.c_attn.weight[..., 1024:2048].reshape(1024, 16, 64).permute(1, 0, 2)
        temp_V = f_model.transformer.h[i].attn.c_attn.weight[..., 2048:3072].reshape(1024, 16, 64).permute(1, 0, 2)

        # attention bias
        temp_Q_b = f_model.transformer.h[i].attn.c_attn.bias[:1024].reshape(16, 64)
        temp_K_b = f_model.transformer.h[i].attn.c_attn.bias[1024:2048].reshape(16, 64)
        temp_V_b = f_model.transformer.h[i].attn.c_attn.bias[2048:3072].reshape(16, 64)

        # mlp weight and bias
        temp_mlp_in = f_model.transformer.h[i].mlp.c_fc.weight
        temp_mlp_out = f_model.transformer.h[i].mlp.c_proj.weight
        temp_mlp_in_bias = f_model.transformer.h[i].mlp.c_fc.bias
        temp_mlp_out_bias = f_model.transformer.h[i].mlp.c_proj.bias

        # layernorm
        temp_ln1_w = f_model.transformer.h[i].ln_1.weight
        temp_ln1_b = f_model.transformer.h[i].ln_1.bias
        temp_ln2_w = f_model.transformer.h[i].ln_2.weight
        temp_ln2_b = f_model.transformer.h[i].ln_2.bias

        # copy
        model.W_Q[i].copy_(temp_Q)
        model.W_K[i].copy_(temp_K)
        model.W_V[i].copy_(temp_V)

        model.b_Q[i].copy_(temp_Q_b)
        model.b_K[i].copy_(temp_K_b)
        model.b_V[i].copy_(temp_V_b)

        model.W_in[i].copy_(temp_mlp_in)
        model.W_out[i].copy_(temp_mlp_out)
        model.b_in[i].copy_(temp_mlp_in_bias)
        model.b_out[i].copy_(temp_mlp_out_bias)

        model.blocks[i].ln1.w = temp_ln1_w
        model.blocks[i].ln1.b = temp_ln1_b
        model.blocks[i].ln2.w = temp_ln2_w
        model.blocks[i].ln2.b = temp_ln2_b

        are_close = []
        are_close.append(t.allclose(model.W_Q[i], temp_Q, atol=1e-0))
        are_close.append(t.allclose(model.W_K[i], temp_K, atol=1e-0))
        are_close.append(t.allclose(model.W_V[i], temp_V, atol=1e-0))
        are_close.append(t.allclose(model.b_Q[i], temp_Q_b, atol=1e-0))
        are_close.append(t.allclose(model.b_K[i], temp_K_b, atol=1e-0))
        are_close.append(t.allclose(model.b_V[i], temp_V_b, atol=1e-0))
        are_close.append(t.allclose(model.W_in[i], temp_mlp_in, atol=1e-0))
        are_close.append(t.allclose(model.W_out[i], temp_mlp_out, atol=1e-0))
        are_close.append(t.allclose(model.b_in[i], temp_mlp_in_bias, atol=1e-0))
        are_close.append(t.allclose(model.b_out[i], temp_mlp_out_bias, atol=1e-0))
        are_close.append(t.allclose(model.blocks[i].ln1.w, temp_ln1_w, atol=1e-0))
        are_close.append(t.allclose(model.blocks[i].ln1.b, temp_ln1_b, atol=1e-0))
        are_close.append(t.allclose(model.blocks[i].ln2.w, temp_ln2_w, atol=1e-0))
        are_close.append(t.allclose(model.blocks[i].ln2.b, temp_ln2_b, atol=1e-0))
        print(f"Are the tensors close?(layer {i}) {are_close}")

    # embedding W_E and W_U
    model.W_U.copy_(f_model.transformer.wte.weight.T)
    t.allclose(model.W_E, model.W_U.T, atol=1e-0)
    print(f"Are the tensors close?(embedding W_E and W_U) {are_close}")

    # layer norm
    model.ln_final.w.copy_(f_model.transformer.ln_f.weight)
    model.ln_final.b.copy_(f_model.transformer.ln_f.bias)

    t.allclose(model.ln_final.w, f_model.transformer.ln_f.weight, atol=1e-0), t.allclose(model.ln_final.b, f_model.transformer.ln_f.bias, atol=1e-0)
    print(f"Are the tensors close?(layer norm) {are_close}")

    return model

## Metric
def metric_denoising( logits, answer_tokens, corrupted_logit_diff, clean_logit_diff):
    patched_logit_diff = compute_logit_diff(logits, answer_tokens)
    patching_effect = (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff  - corrupted_logit_diff) #if positive, patch is better than corrupted, if negative, patch is worse than corrupted
    return patching_effect









def build_label_cache(labels, tokenizer):
    """Tokenize labels once."""
    label_token_ids = [tokenizer.encode(l, add_special_tokens=False) for l in labels]
    return label_token_ids



@t.inference_mode()
def score_label_for_prompt(model, prompt_ids, label_ids, device):
    input_ids = t.tensor([prompt_ids + label_ids], device=device)

    with t.no_grad():
        outputs = model(input_ids)
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, (tuple, list)):
            logits = outputs[0]
        else:
            logits = outputs   # already a tensor

    prompt_len = len(prompt_ids)
    label_len  = len(label_ids)

    logits_to_score = logits[0, prompt_len-1 : prompt_len-1 + label_len, :]
    log_probs = F.log_softmax(logits_to_score, dim=-1)

    label_token_tensor = t.tensor(label_ids, device=device)
    token_log_probs = log_probs.gather(1, label_token_tensor.unsqueeze(1)).squeeze(1)

    return float(token_log_probs.sum().item())

def predict_label(prompt, model, tokenizer, labels, label_token_ids, device):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    scores = [
        score_label_for_prompt(model, prompt_ids, lab_ids, device)
        for lab_ids in label_token_ids
    ]
    best_idx = int(t.tensor(scores).argmax().item())
    return labels[best_idx], scores

def evaluate_accuracy(prompts, labels, model, device):
    tokenizer = model.tokenizer
    label_token_ids = build_label_cache(labels, tokenizer)

    correct = 0
    for prompt, gold in zip(prompts,labels):
        pred, _ = predict_label(prompt, model, tokenizer, labels, label_token_ids, device)
        if pred.strip().lower() == gold.strip().lower():
            correct += 1
        else:
            print("False")
            print(f"Predicted: {pred!r}, Actual: {gold!r}")
            print("Original Sentence: " + prompt + " " + gold)
    return correct / len(labels)