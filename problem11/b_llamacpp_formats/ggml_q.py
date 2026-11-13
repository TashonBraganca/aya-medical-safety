"""llama.cpp (ggml) reference quantizers Q8_0 and Q4_0, written in torch, round trip only (quantize then dequantize).
Ports of quantize_row_q8_0_ref / quantize_row_q4_0_ref and their dequantizers: blocks of 32 along each row, scale stored in fp16.
Checked bit-exact against gguf-py's numpy ports by check_ggml_q.py before use."""
import torch
def q8_0(W):
    x = W.float().reshape(-1, 32); amax = x.abs().amax(1, keepdim=True); d = amax / 127.0
    idv = torch.where(d != 0, 1.0 / d, torch.zeros_like(d)); q = torch.round(x * idv)          # roundf: half away from zero
    q = torch.where((x * idv - torch.trunc(x * idv)).abs() == 0.5, torch.trunc(x * idv) + torch.sign(x * idv), q)
    return (q * d.half().float()).reshape(W.shape)
def q4_0(W):
    x = W.float().reshape(-1, 32); i = x.abs().argmax(1, keepdim=True); mx = x.gather(1, i); d = mx / -8.0
    idv = torch.where(d != 0, 1.0 / d, torch.zeros_like(d)); q = torch.clamp(torch.trunc(x * idv + 8.5), max=15)
    return ((q - 8.0) * d.half().float()).reshape(W.shape)
