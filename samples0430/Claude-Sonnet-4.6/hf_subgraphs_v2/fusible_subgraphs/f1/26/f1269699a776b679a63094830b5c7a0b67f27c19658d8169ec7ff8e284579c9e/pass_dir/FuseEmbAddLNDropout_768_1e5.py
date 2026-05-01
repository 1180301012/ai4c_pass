import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'num_warps': 2}),
        triton.Config({'num_warps': 4}),
        triton.Config({'num_warps': 8}),
        triton.Config({'num_warps': 16}),
    ],
    key=[],
)
@triton.jit
def fused_emb_add_ln_768_kernel(
    word_ids_ptr,
    pos_ids_ptr,
    word_emb_ptr,
    pos_emb_ptr,
    ln_w_ptr,
    ln_b_ptr,
    out_ptr,
    num_tokens,
    H: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_H: tl.constexpr,
    DTYPE: tl.constexpr,  # 0=float32, 1=float16, 2=bfloat16
):
    pid = tl.program_id(0)

    word_idx = tl.load(word_ids_ptr + pid)
    pos_idx = tl.load(pos_ids_ptr + pid)

    h_range = tl.arange(0, BLOCK_H)
    mask = h_range < H

    # Load embeddings and upcast to float32 for numerical stability
    word_emb = tl.load(word_emb_ptr + word_idx * H + h_range, mask=mask, other=0.0).to(tl.float32)
    pos_emb = tl.load(pos_emb_ptr + pos_idx * H + h_range, mask=mask, other=0.0).to(tl.float32)

    # Add embeddings
    x = word_emb + pos_emb

    # Layer norm: compute mean
    x_masked = tl.where(mask, x, 0.0)
    mean = tl.sum(x_masked, axis=0) / H

    # Layer norm: compute variance
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / H

    # Normalize
    rstd = 1.0 / tl.sqrt(var + eps)
    x_norm = (x - mean) * rstd

    # Apply affine transform
    ln_w = tl.load(ln_w_ptr + h_range, mask=mask, other=0.0).to(tl.float32)
    ln_b = tl.load(ln_b_ptr + h_range, mask=mask, other=0.0).to(tl.float32)

    out = x_norm * ln_w + ln_b

    # Store with appropriate dtype cast
    if DTYPE == 2:
        out_typed = out.to(tl.bfloat16)
    elif DTYPE == 1:
        out_typed = out.to(tl.float16)
    else:
        out_typed = out

    tl.store(out_ptr + pid * H + h_range, out_typed, mask=mask)


@torch.fx.wrap
def fused_emb_add_ln_768_1e5(in_0, in_1, in_2, in_3, in_4, in_5):
    # in_0: word_ids  [B, S]       int64
    # in_1: ln_bias   [768]        float
    # in_2: ln_weight [768]        float
    # in_3: pos_emb   [vocab, 768] float
    # in_4: word_emb  [vocab, 768] float
    # in_5: pos_ids   [B, S]       int64
    H = 768
    device = in_4.device
    dtype = in_4.dtype
    shape = in_0.shape
    num_tokens = in_0.numel()

    if device.type == 'cuda':
        word_ids = in_0.view(-1)
        pos_ids = in_5.view(-1).to(device=device)
        pos_emb = in_3.to(device=device)
        ln_w = in_2.to(device=device)
        ln_b = in_1.to(device=device)

        out = torch.empty((num_tokens, H), dtype=dtype, device=device)

        dtype_id = 2 if dtype == torch.bfloat16 else (1 if dtype == torch.float16 else 0)

        fused_emb_add_ln_768_kernel[(num_tokens,)](
            word_ids, pos_ids,
            in_4, pos_emb,
            ln_w, ln_b,
            out,
            num_tokens=num_tokens,
            H=H,
            eps=1e-5,
            BLOCK_H=1024,
            DTYPE=dtype_id,
        )

        return out.view(*shape, H)
    else:
        # CPU fallback using numpy
        import numpy as _np
        word_ids_np = in_0.numpy().flatten()
        pos_ids_np = in_5.numpy().flatten()
        word_emb_np = in_4.numpy().astype(_np.float32)
        pos_emb_np = in_3.numpy().astype(_np.float32)
        ln_w_np = in_2.numpy().astype(_np.float32)
        ln_b_np = in_1.numpy().astype(_np.float32)

        x = word_emb_np[word_ids_np] + pos_emb_np[pos_ids_np]
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        x_norm = (x - mean) / _np.sqrt(var + 1e-5)
        out_np = (x_norm * ln_w_np + ln_b_np).reshape(*shape, H)

        return torch.as_tensor(out_np.astype(_np.float32 if dtype == torch.float32 else _np.float16), dtype=dtype)


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    tmp_5 = torch.nn.functional.embedding(in_0, in_4, 1, None, 2.0, False, False)
    tmp_6 = torch.nn.functional.embedding(in_5, in_3, 1, None, 2.0, False, False)
    tmp_7 = tmp_5 + tmp_6
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (768,), in_2, in_1, 1e-05)
    tmp_9 = torch.nn.functional.dropout(tmp_8, 0.1, False, False)
    return tmp_9


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


def replacement_func():
    return fused_emb_add_ln_768_1e5