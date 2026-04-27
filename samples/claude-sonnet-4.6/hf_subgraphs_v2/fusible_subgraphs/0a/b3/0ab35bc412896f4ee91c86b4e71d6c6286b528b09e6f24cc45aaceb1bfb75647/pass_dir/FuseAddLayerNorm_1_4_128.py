import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_3 + in_2
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (128,), in_1, in_0, 1e-05)
    tmp_4 = tmp_3.reshape(1, 2, 2, -1)
    tmp_5 = tmp_4.permute(0, 3, 1, 2)
    tmp_6 = tmp_5.contiguous()
    tmp_7 = tmp_6.permute(0, 2, 3, 1)
    tmp_8 = tmp_7.reshape(1, -1, 128)
    return tmp_8


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.jit
def _fused_add_layernorm_kernel(
    x_ptr,
    y_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    C: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # One program per row (B*N rows total)
    row_id = tl.program_id(0)
    row_start = row_id * C
    cols = tl.arange(0, BLOCK_C)

    # Load both inputs and fuse-add (promote to fp32 for numeric accuracy)
    x = tl.load(x_ptr + row_start + cols)
    y = tl.load(y_ptr + row_start + cols)
    val = x.to(tl.float32) + y.to(tl.float32)

    # Mean
    mean = tl.sum(val, axis=0) / C

    # Variance
    diff = val - mean
    var = tl.sum(diff * diff, axis=0) / C

    # Normalize
    inv_std = tl.rsqrt(var + eps)
    norm = diff * inv_std

    # Scale and shift
    w = tl.load(w_ptr + cols).to(tl.float32)
    b_val = tl.load(b_ptr + cols).to(tl.float32)
    result = norm * w + b_val

    # Store contiguously — values are identical to LN output
    tl.store(out_ptr + row_start + cols, result.to(x.dtype))


@torch.fx.wrap
def _fused_add_layernorm_wrapper(in_0, in_1, in_2, in_3):
    # in_0: bias  [C]
    # in_1: weight [C]
    # in_2, in_3: [B, N, C]  (contiguous, float16 or bfloat16)
    B, N, C = in_2.shape
    out = torch.empty(B, N, C, dtype=in_2.dtype, device=in_2.device)

    total_rows = B * N
    _fused_add_layernorm_kernel[(total_rows,)](
        in_3,        # x  (in_3 + in_2)
        in_2,        # y
        in_1,        # weight
        in_0,        # bias
        out,
        C=128,
        eps=1e-5,
        BLOCK_C=128,
        num_warps=4,
    )
    return out


def replacement_func():
    return _fused_add_layernorm_wrapper