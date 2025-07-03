import torch as t
from torch import Tensor
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
from tqdm import tqdm
from functools import partial
import numpy as np

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

def compute_logit_diff(logits, answer_tokens, array=False):
    last = logits[:, -1, :]
    answer_logits = last.gather(dim=-1, index=answer_tokens)
    correct_logits, incorrect_logits = answer_logits.unbind(dim=-1)
    answer_logit_diff = correct_logits - incorrect_logits

    if array:
        return answer_logit_diff.cpu().numpy()
        
    return answer_logit_diff.mean()

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
    patching_effect = (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff  - corrupted_logit_diff)
    return patching_effect

## Intervention
def patching_residual_hook(corrupted_residual_component, hook, pos, clean_cache):
    corrupted_residual_component[:, pos, :] = clean_cache[hook.name][:, pos, :]
    return corrupted_residual_component

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

def patching_attention_hook(corrupted_head_vector, hook, head_index, clean_cache):
    corrupted_head_vector[:, :, head_index] = clean_cache[hook.name][:, :, head_index]
    return corrupted_head_vector

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




