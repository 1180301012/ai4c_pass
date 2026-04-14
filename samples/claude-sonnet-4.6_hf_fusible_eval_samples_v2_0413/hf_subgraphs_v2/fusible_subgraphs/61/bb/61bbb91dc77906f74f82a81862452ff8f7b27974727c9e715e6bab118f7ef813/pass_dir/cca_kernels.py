"""
Shared Triton kernels and single dispatch wrapper for all CCA-related passes.
Both OptimizedEinsum.py and FusedCCAEinsumScaleAdd.py import cca_dispatch from
here so that replacement_func() is identical across all passes, satisfying the
output_pass_replacement_func_limit requirement.
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: bchj,bhwj->bchw  batched GEMM
# For each (b, h):  out[b,:,h,:] = in4[b,:,h,:] @ in1[b,h,:,:].T
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 16},  num_warps=4,  num_stages=4),
        triton.Config({'BLOCK_C': 32},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_C': 32},  num_warps=4,  num_stages=4),
        triton.Config({'BLOCK_C': 32},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_C': 64},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_C': 64},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_C': 64},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_C': 128}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_C': 128}, num_warps=16, num_stages=2),
    ],
    key=['B', 'C', 'H'],
)
@triton.jit
def einsum_kernel(
    in4_ptr, in1_ptr, out_ptr,
    B, C, H, W, J,
    s4_b, s4_c, s4_h, s4_j,
    s1_b, s1_h, s1_w, s1_j,
    so_b, so_c, so_h, so_w,
    IS_BF16: tl.constexpr,
    IS_FP16: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_W: tl.constexpr,   # == W (64)
    BLOCK_J: tl.constexpr,   # == J (64)
):
    pid_bh = tl.program_id(0)
    pid_c  = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh %  H

    c_start = pid_c * BLOCK_C
    c_offs  = c_start + tl.arange(0, BLOCK_C)
    w_offs  = tl.arange(0, BLOCK_W)
    j_offs  = tl.arange(0, BLOCK_J)
    c_mask  = c_offs < C

    # Load in native dtype so tl.dot can use fp16/bf16 tensor cores directly
    in4_base = b * s4_b + h * s4_h
    in4_vals = tl.load(
        in4_ptr + in4_base + c_offs[:, None] * s4_c + j_offs[None, :] * s4_j,
        mask=c_mask[:, None], other=0.0).to(tl.float32)

    in1_base = b * s1_b + h * s1_h
    in1_vals = tl.load(
        in1_ptr + in1_base + w_offs[:, None] * s1_w + j_offs[None, :] * s1_j).to(tl.float32)

    # fp32 tl.dot — accumulation in fp32 for precision + compatibility
    acc = tl.dot(in4_vals, tl.trans(in1_vals))

    valid   = c_mask[:, None]
    out_off = b * so_b + h * so_h + c_offs[:, None] * so_c + w_offs[None, :] * so_w
    if IS_BF16:
        tl.store(out_ptr + out_off, acc.to(tl.bfloat16), mask=valid)
    elif IS_FP16:
        tl.store(out_ptr + out_off, acc.to(tl.float16),  mask=valid)
    else:
        tl.store(out_ptr + out_off, acc,                  mask=valid)


# ---------------------------------------------------------------------------
# Kernel 2: fused  in_5 * scalar + in_2  (scale + residual, flat 1-D)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK': 2048},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK': 4096},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 8192},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 16384}, num_warps=8, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def scale_add_kernel(
    in5_ptr, in0_ptr, in2_ptr, out_ptr,
    N,
    IS_BF16: tl.constexpr,
    IS_FP16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    val   = tl.load(in5_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(in0_ptr).to(tl.float32)
    res   = tl.load(in2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    result = val * scale + res
    if IS_BF16:
        tl.store(out_ptr + offs, result.to(tl.bfloat16), mask=mask)
    elif IS_FP16:
        tl.store(out_ptr + offs, result.to(tl.float16),  mask=mask)
    else:
        tl.store(out_ptr + offs, result,                  mask=mask)


# ---------------------------------------------------------------------------
# Single shared dispatch wrapper (returned by replacement_func() in all passes)
#
# route="einsum":    arg0=in_4 [B,C,H,J], arg1=in_1 [B,H,W,J], arg2=None
# route="scale_add": arg0=in_0 (scalar),  arg1=in_2 [B,C,H,W], arg2=in_5 [B,C,H,W]
# ---------------------------------------------------------------------------

@torch.fx.wrap
def cca_dispatch(arg0, arg1, arg2, route):
    if route == "einsum":
        in_4, in_1 = arg0, arg1
        B, C, H, J = in_4.shape
        W = in_1.shape[2]
        out = torch.empty(B, C, H, W, dtype=in_4.dtype, device=in_4.device)
        IS_BF16 = (in_4.dtype == torch.bfloat16)
        IS_FP16 = (in_4.dtype == torch.float16)
        grid = lambda meta: (B * H, (C + meta['BLOCK_C'] - 1) // meta['BLOCK_C'])
        einsum_kernel[grid](
            in_4, in_1, out,
            B, C, H, W, J,
            in_4.stride(0), in_4.stride(1), in_4.stride(2), in_4.stride(3),
            in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
            out.stride(0),  out.stride(1),  out.stride(2),  out.stride(3),
            IS_BF16=IS_BF16, IS_FP16=IS_FP16,
            BLOCK_W=W, BLOCK_J=J,
        )
        return out
    else:  # "scale_add"
        in_0, in_2, in_5 = arg0, arg1, arg2
        N = in_5.numel()
        out = torch.empty_like(in_5)
        IS_BF16 = (in_5.dtype == torch.bfloat16)
        IS_FP16 = (in_5.dtype == torch.float16)
        grid = lambda meta: ((N + meta['BLOCK'] - 1) // meta['BLOCK'],)
        scale_add_kernel[grid](
            in_5, in_0, in_2, out,
            N,
            IS_BF16=IS_BF16, IS_FP16=IS_FP16,
        )
        return out