"""THIN WRAPPER (our code) that lets CircuitKIT's ActivationSteering steer GENERATION.

CircuitKIT's `ActivationSteering.steer()` registers its own steering hook functions with `model.add_hook(hook_point, hook_fn)`,
runs ONE forward pass, and removes them (`reset_hooks`). It has no generate(). This module:
  1. capture_ck_hooks: calls steer() once on a short dummy input while recording the exact (hook_point, hook_fn) pairs
     CircuitKIT registers (a spy around model.add_hook that forwards every call unchanged);
  2. tl_generate: keeps exactly those CircuitKIT hook functions installed (`model.hooks(fwd_hooks=...)`, TransformerLens) while
     TransformerLens' own HookedTransformer.generate runs (greedy, KV cache, left padding, attention mask).
CircuitKIT code: hook-point choice (_get_hook_point_from_node), the steering arithmetic (hook_result[:, :, head] += c*v), the TL port.
TransformerLens code: the generation loop. Ours: hook capture and lifetime, batching, tokenisation, decoding."""
import torch


def prepare_for_generation(tl):
    """ck.load_model turns on four EAP flags. Two of them (ungroup_grouped_query_attention, use_split_qkv_input) are needed only by
    EAP and make TransformerLens 3.8's KV-cached generate crash on GQA models (cache sized for n_key_value_heads, ungrouped keys
    have n_heads: 'Expected size 2 but got size 4'). Turning them off is the same arithmetic (tiny Cohere2: max logit diff 8.8e-7)
    and fixes it. use_attn_result stays on: CircuitKIT's steering hooks live on attn.hook_result."""
    tl.cfg.ungroup_grouped_query_attention = False
    tl.set_use_split_qkv_input(False)
    tl.set_use_attn_result(True)
    return tl


def capture_ck_hooks(st, vecs, coeff, dummy_ids):
    """Run CircuitKIT's st.steer(dummy, vecs, coeff) once; return the list of (hook_point, hook_fn) CircuitKIT registered."""
    model = st.model; rec = []
    orig = model.add_hook                      # bound method of the HookedTransformer
    def spy(name, hook, *args, **kw):
        rec.append((name, hook)); return orig(name, hook, *args, **kw)
    model.add_hook = spy                       # instance attribute shadows the method for this call only
    try:
        res = st.steer(dummy_ids, steering_vectors=vecs, coefficient=coeff)
    finally:
        del model.add_hook
    assert "add_hook" not in model.__dict__
    assert len(rec) == len(vecs) == len(res["steered_nodes"]), (len(rec), len(vecs), res["steered_nodes"])
    return rec


@torch.no_grad()
def tl_generate(tl, prompt_ids, max_new, eos, pad_id, hooks=(), bs=48):
    """Greedy generation with TransformerLens' generate. prompt_ids: list of token-id lists (chat template incl. BOS).
    hooks: CircuitKIT hook functions from capture_ck_hooks (kept installed for the whole generation). Returns list of
    {"ids": new token ids up to (excluding) the first eos, "n_tokens", "hit_max"} in input order."""
    out = [None] * len(prompt_ids); order = sorted(range(len(prompt_ids)), key=lambda i: -len(prompt_ids[i]))
    tl.tokenizer.padding_side = "left"
    dev = tl.cfg.device
    for k in range(0, len(order), bs):
        idx = order[k:k + bs]; L = max(len(prompt_ids[i]) for i in idx)
        ids = torch.full((len(idx), L), pad_id, dtype=torch.long)
        for r, i in enumerate(idx): ids[r, L - len(prompt_ids[i]):] = torch.tensor(prompt_ids[i])
        with tl.hooks(fwd_hooks=list(hooks)):
            g = tl.generate(ids.to(dev), max_new_tokens=max_new, do_sample=False, stop_at_eos=True, eos_token_id=list(eos),
                            use_past_kv_cache=True, prepend_bos=False, padding_side="left", return_type="tokens", verbose=False)
        assert not any(len(hp.fwd_hooks) for hp in tl.hook_dict.values()), "hooks leaked after generation"
        for i, row in zip(idx, g[:, L:].tolist()):
            n = next((j for j, t in enumerate(row) if t in eos), len(row))
            out[i] = {"ids": row[:n], "n_tokens": n, "hit_max": n >= max_new}
    return out


@torch.no_grad()
def tl_seq_lp(forward, items, pad_id, bs=4):
    """mean log p of the last n tokens for each (ids, n); right-padded batches. forward(ids) -> logits [B, T, V].
    No attention mask: causal attention means trailing pads never affect earlier (real) positions."""
    import math
    out = [None] * len(items); order = sorted(range(len(items)), key=lambda i: -len(items[i][0]))
    for k in range(0, len(order), bs):
        idx = order[k:k + bs]; T = len(items[idx[0]][0])
        ids = torch.full((len(idx), T), pad_id, dtype=torch.long)
        for j, i in enumerate(idx): ids[j, :len(items[i][0])] = torch.tensor(items[i][0])
        lg = forward(ids)
        for j, i in enumerate(idx):
            s, n = items[i]; L = len(s)
            lp = torch.log_softmax(lg[j, L - n - 1:L - 1].float(), -1); tgt = torch.tensor(s[L - n:], device=lp.device)
            out[i] = float(lp[torch.arange(n, device=lp.device), tgt].mean())
        del lg
    assert all(x is not None and math.isfinite(x) for x in out)
    return out
