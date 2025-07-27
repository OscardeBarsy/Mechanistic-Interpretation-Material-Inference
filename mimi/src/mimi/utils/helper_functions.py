import torch
import torch.nn.functional as F
from torch import Tensor
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
from tqdm import tqdm
from functools import partial
import numpy as np

## Misc
def normalise_tensor(tensor):
    max_abs_val = torch.max(torch.abs(tensor))
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
def compute_logit_diff(logits, answer_tokens, array=False):
    last = logits[:, -1, :]
    answer_logits = last.gather(dim=-1, index=answer_tokens)
    correct_logits, incorrect_logits = answer_logits.unbind(dim=-1)
    answer_logit_diff = correct_logits - incorrect_logits

    if array:
        return answer_logit_diff.cpu().numpy()
        
    return answer_logit_diff.mean()

def compute_logit_diff_seqavg(logits_or_last, answer_token_seqs, array=False):
    """
    Accepts:
      logits_or_last: [B, T, V] or [B, V] (CUDA or CPU)
      answer_token_seqs: list of length B of (corr_ids_1D, inc_ids_1D), CPU Long
    Returns:
      mean diff (Tensor scalar) or per-example numpy if array=True
    """
    # Get last-step logits and move to CPU
    if logits_or_last.dim() == 3:
        last = logits_or_last[:, -1, :]
    elif logits_or_last.dim() == 2:
        last = logits_or_last
    else:
        raise ValueError(f"Expected [B,T,V] or [B,V], got shape {tuple(logits_or_last.shape)}")
    last = last.detach()
    if last.is_cuda:
        last = last.cpu()

    B, V = last.shape
    diffs = []

    for i, (corr_ids, inc_ids) in enumerate(answer_token_seqs):
        corr_ids = torch.as_tensor(corr_ids, dtype=torch.long, device="cpu").view(-1)
        inc_ids  = torch.as_tensor(inc_ids,  dtype=torch.long, device="cpu").view(-1)

        # Early sanity checks (raise a clear error instead of a CUDA assert)
        if corr_ids.numel() == 0 or inc_ids.numel() == 0:
            raise ValueError(f"Empty token list at batch index {i}: "
                             f"corr_len={corr_ids.numel()}, inc_len={inc_ids.numel()}")
        if int(corr_ids.max()) >= V or int(inc_ids.max()) >= V or int(corr_ids.min()) < 0 or int(inc_ids.min()) < 0:
            raise ValueError(
                f"Token id out of vocab range at batch index {i}: "
                f"max_corr={int(corr_ids.max())}, max_inc={int(inc_ids.max())}, V={V}"
            )

        corr_mean = last[i, corr_ids].mean()
        inc_mean  = last[i, inc_ids].mean()
        diffs.append(corr_mean - inc_mean)

    diffs = torch.stack(diffs)  # [B]
    return diffs.numpy() if array else diffs.mean()




def get_batched_logit_diff(minibatch_size, tokens, answer_token_seqs, model):
    avg_logit_diff = 0
    for i in range(0, len(tokens), minibatch_size):
        logit, _ = model.run_with_cache(tokens[i : i+minibatch_size])
        diffs = compute_logit_diff_seqavg(logit, answer_token_seqs[i : i+minibatch_size])
        avg_logit_diff += diffs
        del logit, diffs
    return avg_logit_diff / (len(tokens) / minibatch_size)


def align_fine_tuning_gpt2(model, f_model, device):
    model.W_E.copy_(f_model.transformer.wte.weight)
    are_close = torch.allclose(model.W_E, f_model.transformer.wte.weight)
    print(f"Are the tensors close?(token embedding) {are_close}")

    model.W_pos.copy_(f_model.transformer.wpe.weight)
    are_close = torch.allclose(model.W_pos, f_model.transformer.wpe.weight)
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
        are_close.append(torch.allclose(model.W_Q[i], temp_Q, atol=1e-0))
        are_close.append(torch.allclose(model.W_K[i], temp_K, atol=1e-0))
        are_close.append(torch.allclose(model.W_V[i], temp_V, atol=1e-0))
        are_close.append(torch.allclose(model.b_Q[i], temp_Q_b, atol=1e-0))
        are_close.append(torch.allclose(model.b_K[i], temp_K_b, atol=1e-0))
        are_close.append(torch.allclose(model.b_V[i], temp_V_b, atol=1e-0))
        are_close.append(torch.allclose(model.W_in[i], temp_mlp_in, atol=1e-0))
        are_close.append(torch.allclose(model.W_out[i], temp_mlp_out, atol=1e-0))
        are_close.append(torch.allclose(model.b_in[i], temp_mlp_in_bias, atol=1e-0))
        are_close.append(torch.allclose(model.b_out[i], temp_mlp_out_bias, atol=1e-0))
        are_close.append(torch.allclose(model.blocks[i].ln1.w, temp_ln1_w, atol=1e-0))
        are_close.append(torch.allclose(model.blocks[i].ln1.b, temp_ln1_b, atol=1e-0))
        are_close.append(torch.allclose(model.blocks[i].ln2.w, temp_ln2_w, atol=1e-0))
        are_close.append(torch.allclose(model.blocks[i].ln2.b, temp_ln2_b, atol=1e-0))
        print(f"Are the tensors close?(layer {i}) {are_close}")

    # embedding W_E and W_U
    model.W_U.copy_(f_model.transformer.wte.weight.T)
    torch.allclose(model.W_E, model.W_U.T, atol=1e-0)
    print(f"Are the tensors close?(embedding W_E and W_U) {are_close}")

    # layer norm
    model.ln_final.w.copy_(f_model.transformer.ln_f.weight)
    model.ln_final.b.copy_(f_model.transformer.ln_f.bias)

    torch.allclose(model.ln_final.w, f_model.transformer.ln_f.weight, atol=1e-0), torch.allclose(model.ln_final.b, f_model.transformer.ln_f.bias, atol=1e-0)
    print(f"Are the tensors close?(layer norm) {are_close}")

    return model

## Metric
def metric_denoising(logits_or_last, answer_token_seqs, corrupted_logit_diff, clean_logit_diff):
    # Accept [B,T,V] or [B,V], and do indexing on CPU
    if logits_or_last.dim() == 3:
        last = logits_or_last[:, -1, :].detach()
    elif logits_or_last.dim() == 2:
        last = logits_or_last.detach()
    else:
        raise ValueError(f"Expected [B,T,V] or [B,V], got {tuple(logits_or_last.shape)}")

    if last.is_cuda:
        last = last.cpu()
    patched_logit_diff = compute_logit_diff_seqavg(last, answer_token_seqs)
    return (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)


## Intervention
# ---- minibatched residual hook ----
def patching_residual_hook_batch(corrupted_resid, hook, pos, clean_cache, batch_slice):
    s, e = batch_slice
    # corrupted_resid: [mb, seq_len, d_model]
    # clean_cache[hook.name]: [B, seq_len, d_model]
    corrupted_resid[:, pos, :] = clean_cache[hook.name][s:e, pos, :]
    return corrupted_resid

# ---- minibatched residual patch driver ----
def patching_residual(model, corrupted_tokens, clean_cache, patching_metric,
                      answer_token_seqs, clean_logit_diff, corrupted_logit_diff,
                      device, minibatch_size=16):
    model.reset_hooks()
    B, T = corrupted_tokens.size(0), corrupted_tokens.size(1)
    results = torch.zeros(model.cfg.n_layers, T, device=device, dtype=torch.float32)

    spans = [(i, min(i+minibatch_size, B)) for i in range(0, B, minibatch_size)]

    for layer in range(model.cfg.n_layers):
        for pos in range(T):
            accum = 0.0
            for (s, e) in spans:
                hook_fn = partial(
                    patching_residual_hook_batch,
                    pos=pos,
                    clean_cache=clean_cache,
                    batch_slice=(s, e),
                )
                # forward only this chunk
                patched_logits = model.run_with_hooks(
                    corrupted_tokens[s:e],
                    fwd_hooks=[(utils.get_act_name("resid_post", layer), hook_fn)],
                    return_type="logits",
                )

                # free GPU ASAP: keep only last-step logits on CPU
                last = patched_logits[:, -1, :].detach().cpu()
                del patched_logits
                torch.cuda.empty_cache()

                # metric_denoising accepts [mb,V] or [mb,T,V]
                chunk_metric = patching_metric(
                    last, answer_token_seqs[s:e], corrupted_logit_diff, clean_logit_diff
                )


def patching_attention_hook(corrupted_head_vector, hook, head_index, clean_cache):
    corrupted_head_vector[:, :, head_index] = clean_cache[hook.name][:, :, head_index]
    return corrupted_head_vector

def patching_attention_hook_batch(corrupted_head_vector, hook, head_index, clean_cache, batch_slice):
    s, e = batch_slice
    # corrupted_head_vector: [mb, seq_len, n_heads, d_head] for this layer
    # clean_cache[hook.name]: [B,  seq_len, n_heads, d_head]
    corrupted_head_vector[:, :, head_index] = clean_cache[hook.name][s:e, :, head_index]
    return corrupted_head_vector

def patching_attention(model, corrupted_tokens, clean_cache, patching_metric,
                       answer_token_seqs, clean_logit_diff, corrupted_logit_diff,
                       head_type, device, minibatch_size=16):
    """
    head_type: usually 'z' (result), 'pattern', etc.
    minibatch_size: tune to fit your GPU (e.g., 8/16/32)
    """
    model.reset_hooks()
    results = torch.zeros(model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=torch.float32)

    B = corrupted_tokens.size(0)
    spans = [(i, min(i + minibatch_size, B)) for i in range(0, B, minibatch_size)]

    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            accum = 0.0
            for (s, e) in spans:
                hook_fn = partial(
                    patching_attention_hook_batch,
                    head_index=head,
                    clean_cache=clean_cache,
                    batch_slice=(s, e),
                )
                # Forward only this chunk
                patched_logits = model.run_with_hooks(
                    corrupted_tokens[s:e],
                    fwd_hooks=[(utils.get_act_name(head_type, layer), hook_fn)],
                    return_type="logits",
                )

                # Reduce memory ASAP: go to last step on CPU here
                last = patched_logits[:, -1, :].detach().cpu()
                del patched_logits
                torch.cuda.empty_cache()

                # metric_denoising accepts [mb,V] now
                chunk_metric = patching_metric(last, answer_token_seqs[s:e], corrupted_logit_diff, clean_logit_diff)
                accum += float(chunk_metric) * (e - s)

            results[layer, head] = accum / B

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
    results = torch.zeros(model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=torch.float32)

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

    return compute_logit_diff_seqavg(patched_logits, answer_tokens)

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



def build_label_cache(labels, tokenizer):
    """Tokenize labels once."""
    label_token_ids = [tokenizer.encode(l, add_special_tokens=False) for l in labels]
    return label_token_ids

@torch.inference_mode()
def score_label_for_prompt(model, tokenizer, prompt_ids, label_ids, device):
    input_ids = torch.tensor([prompt_ids + label_ids], device=device)

    with torch.no_grad():
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

    label_token_tensor = torch.tensor(label_ids, device=device)
    token_log_probs = log_probs.gather(1, label_token_tensor.unsqueeze(1)).squeeze(1)

    return float(token_log_probs.sum().item())

def predict_label(prompt, model, tokenizer, labels, label_token_ids, device):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    scores = [
        score_label_for_prompt(model, tokenizer, prompt_ids, lab_ids, device)
        for lab_ids in label_token_ids
    ]
    best_idx = int(torch.tensor(scores).argmax().item())
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
            print(f"Predicted: {pred!r}, Actual: {gold!r}")
            print("Original Sentence: " + prompt + " " + gold)
    return correct / len(labels)

