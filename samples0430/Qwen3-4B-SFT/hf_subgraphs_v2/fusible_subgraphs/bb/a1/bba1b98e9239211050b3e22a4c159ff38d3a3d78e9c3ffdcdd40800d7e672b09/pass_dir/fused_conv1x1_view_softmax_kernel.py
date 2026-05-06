import torch
import triton
import triton.language as tl


# All configs satisfy BLOCK_C * BLOCK_N <= 65536 so that every intermediate
# 2-D tensor (x: [BLOCK_N, BLOCK_C]) stays within Triton's tensor-size limit.
# The loop over channels is REMOVED as `BLOCK_C` is always equal to `C` (=512).
configs = [
    triton.Config({'BLOCK_N': 1,  'BLOCK_C': 512}, num_warps=2),
    triton.Config({'BLOCK_N': 2,  'BLOCK_C': 512}, num_warps=2),
    triton.Config({'BLOCK_N': 4,  'BLOCK_C': 512}, num_warps=4),
    triton.Config({'BLOCK_N': 8,  'BLOCK_C': 512}, num_warps=4),
    triton.Config({'BLOCK_N': 16, 'BLOCK_C': 512}, num_warps=4),
    triton.Config({'BLOCK_N': 32, 'BLOCK_C': 512}, num_warps=4),
    triton.Config({'BLOCK_N': 64, 'BLOCK_C': 512}, num_warps=4),
    triton.Config({'BLOCK_N': 128,'BLOCK_C': 512}, num_warps=4),
    triton.Config({'BLOCK_N': 256,'BLOCK_C': 512}, num_warps=4),
    triton.Config({'BLOCK_N': 1,  'BLOCK_C': 256}, num_warps=2),
    triton.Config({'BLOCK_N': 2,  'BLOCK_C': 256}, num_warps=2),
    triton.Config({'BLOCK_N': 4,  'BLOCK_C': 256}, num_warps=2),
    triton.Config({'BLOCK_N': 8,  'BLOCK_C': 256}, num_warps=4),
    triton.Config({'BLOCK_N': 16, 'BLOCK_C': 256}, num_warps=4),
    triton.Config({'BLOCK_N': 32, 'BLOCK_C': 256}, num_warps=4),
    triton.Config({'BLOCK_N': 64, 'BLOCK_C': 256}, num_warps=4),
    triton.Config({'BLOCK_N': 128,'BLOCK_C': 256}, num_warps=4),
    triton.Config({'BLOCK_N': 256,'BLOCK_C': 256}, num_warps=4),
]


@triton.autotune(configs, key=['B', 'N', 'C'])
@triton.jit
def fused_conv1x1_softmax_kernel(
    input_ptr,   # [B, C, N]  – contiguous NCHW data
    weight_ptr,  # [C]        – flattened weight  (shape [1,C,1,1])
    bias_ptr,    # [1]
    output_ptr,  # flat [B*N] buffer
    B, C, N,
    BLOCK_F16: tl.constexpr,  # True → fp16 output
    BLOCK_BF16: tl.constexpr, # True → bf16 output
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # No channel loop: BLOCK_C is always a divisor of C (typically C itself).
    # Each kernel program handles one (batch, n_block) tile.
    pid = tl.program_id(0)
    n_block = pid // B
    b = pid % B

    n_start = n_block * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)   # [BLOCK_N]
    n_mask = n_offs < N                          # [BLOCK_N]

    # Weight as [1, BLOCK_C]: w_v[n, c] = weight[c]  (all BLOCK_N rows are identical)
    c_offs = tl.arange(0, BLOCK_C)
    mask_c = c_offs < C
    w_v = tl.load(weight_ptr + c_offs, mask=mask_c, other=0.0)  # [BLOCK_C]
    w_v = w_v[None, :]                              # [1, BLOCK_C]   ← no broadcast_to

    # Input as [BLOCK_N, BLOCK_C]: x[n, c] = input[b, c, n_start+n]
    # Index formula: b*C*N + c*N + n   → reshaped as (b*C*N + n) + c*N = [BLOCK_N, BLOCK_C]
    in_offs = b * C * N + n_offs[:, None] + c_offs[None, :] * N  # [BLOCK_N, BLOCK_C]
    x = tl.load(
        input_ptr + in_offs,
        mask=n_mask[:, None] & mask_c[None, :],
        other=0.0,
    )  # [BLOCK_N, BLOCK_C]

    # Dot product via broadcast × → [BLOCK_N, BLOCK_C] → sum over axis=1 → [BLOCK_N]
    acc = tl.sum(x * w_v, axis=1).to(tl.float32)

    # Add bias (scalar → broadcast over [BLOCK_N])
    acc += tl.load(bias_ptr).to(tl.float32)

    # Softmax
    acc = tl.where(n_mask, acc, float('-inf'))
    max_acc = tl.max(acc, axis=0)
    exp_acc = tl.exp(acc - max_acc)
    exp_acc = tl.where(n_mask, exp_acc, 0.0)
    result = exp_acc / tl.sum(exp_acc, axis=0)

    out_idx = b * N + n_offs
    if BLOCK_BF16:
        tl.store(output_ptr + out_idx, result.to(tl.bfloat16), mask=n_mask)
    elif BLOCK_F16:
        tl.store(output_ptr + out_idx, result.to(tl.float16),  mask=n_mask)
    else:
        tl.store(output_ptr + out_idx, result.to(tl.float32),  mask=n_mask)


@torch.fx.wrap
def fused_conv1x1_view_softmax(bias, weight, x):
    """
    Fused 1×1 conv + view + softmax.

    weight  : [1, C, 1, 1]
    bias    : [1]
    x       : [B, C, H, W]
    returns : [B, 1, H*W]  -- softmax over the N=H×W spatial positions,
               with same dtype as input.
    """
    B = x.shape[0]
    C = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]
    N = H * W

    is_f16  = (x.dtype == torch.float16)
    is_bf16 = (x.dtype == torch.bfloat16)

    # Write everything in fp32 for numerical stability, then cast back.
    out_buf = torch.empty((B * N,), dtype=torch.float32, device=x.device)

    grid = lambda meta: (triton.cdiv(B * N, meta['BLOCK_N']),)

    fused_conv1x1_softmax_kernel[grid](
        x, weight, bias, out_buf,
        B, C, N,
        BLOCK_F16=is_f16,
        BLOCK_BF16=is_bf16,
    )

    # Cast back to the original dtype (tensor method, allowed in FX pass).
    return out_buf.to(x.dtype).view(B, 1, N)