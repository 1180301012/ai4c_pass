import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    tmp_4 = torch.nn.functional.layer_norm(in_3, (768,), in_2, in_1, 1e-12)
    tmp_5 = in_0.unsqueeze(-1)
    tmp_6 = tmp_5.expand_as(tmp_4)
    tmp_7 = tmp_6.float()
    tmp_8 = tmp_4 * tmp_7
    return (tmp_7, tmp_8, tmp_4)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
    ],
    key=['N_COLS'],
)
@triton.jit
def fused_layernorm_mask_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    mask_ptr,
    out_ln_ptr,
    out_mask_ptr,
    out_mul_ptr,
    N_COLS: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < N_COLS

    # Load input row, padding invalid positions with 0.0
    x = tl.load(
        x_ptr + row_idx * N_COLS + col_offsets,
        mask=col_mask,
        other=0.0,
    ).to(tl.float32)

    # Compute mean — padded elements are 0.0, so they don't pollute the sum
    mean = tl.sum(x, axis=0) / N_COLS

    # Compute variance — explicitly zero out padded element contributions
    diff = tl.where(col_mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N_COLS

    # Normalize
    rstd = 1.0 / tl.sqrt(var + eps)
    x_norm = (x - mean) * rstd

    # Load weight and bias (in_2 = weight, in_1 = bias)
    w = tl.load(w_ptr + col_offsets, mask=col_mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + col_offsets, mask=col_mask, other=0.0).to(tl.float32)

    # Apply affine transform — result in float32
    y = x_norm * w + b

    # Load scalar attention mask for this row (int64 → float32)
    mask_val = tl.load(mask_ptr + row_idx).to(tl.float32)

    # ---------------------------------------------------------------
    # Store tmp_4 : layer-norm output converted back to original dtype
    # Triton auto-casts float32 → element dtype of out_ln_ptr
    # ---------------------------------------------------------------
    tl.store(
        out_ln_ptr + row_idx * N_COLS + col_offsets,
        y,
        mask=col_mask,
    )

    # Store tmp_7 : float32 broadcast of mask value
    tl.store(
        out_mask_ptr + row_idx * N_COLS + col_offsets,
        mask_val,
        mask=col_mask,
    )

    # Store tmp_8 : layer-norm * mask  (float32 × float32 → float32)
    tl.store(
        out_mul_ptr + row_idx * N_COLS + col_offsets,
        y * mask_val,
        mask=col_mask,
    )


@torch.fx.wrap
def fused_layernorm_mask(in_0, in_1, in_2, in_3):
    # in_0  : [B, S]       int64  – attention mask
    # in_1  : [768]        fp16/bf16 – LayerNorm bias
    # in_2  : [768]        fp16/bf16 – LayerNorm weight
    # in_3  : [B, S, 768]  fp16/bf16 – hidden states
    orig_shape = in_3.shape          # e.g. [1, 16, 768]
    N_ROWS = orig_shape[0] * orig_shape[1]   # B * S
    N_COLS = orig_shape[-1]                  # 768

    # Flatten spatial dims for Triton (ensure contiguous)
    x_flat   = in_3.contiguous().reshape(N_ROWS, N_COLS)
    mask_flat = in_0.contiguous().reshape(N_ROWS)

    # Allocate flat output buffers
    out_ln   = torch.empty((N_ROWS, N_COLS), dtype=in_3.dtype,      device=in_3.device)
    out_mask = torch.empty((N_ROWS, N_COLS), dtype=torch.float32,   device=in_3.device)
    out_mul  = torch.empty((N_ROWS, N_COLS), dtype=torch.float32,   device=in_3.device)

    fused_layernorm_mask_kernel[(N_ROWS,)](
        x_flat, in_2, in_1, mask_flat,
        out_ln, out_mask, out_mul,
        N_COLS=N_COLS,
        eps=1e-12,
    )

    # Return (tmp_7, tmp_8, tmp_4) — matching the pattern return order
    return (
        out_mask.view(orig_shape),
        out_mul.view(orig_shape),
        out_ln.view(orig_shape),
    )


def replacement_func():
    return fused_layernorm_mask