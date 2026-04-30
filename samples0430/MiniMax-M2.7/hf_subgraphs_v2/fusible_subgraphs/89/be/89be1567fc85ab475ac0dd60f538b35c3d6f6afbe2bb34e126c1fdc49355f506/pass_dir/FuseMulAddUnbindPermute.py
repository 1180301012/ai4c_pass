import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64, 'num_warps': 2}, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128, 'num_warps': 4}, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256, 'num_warps': 4}, num_stages=2),
    ],
    key=['D'],
)
@triton.jit
def triton_fused_kernel(
    in_2_ptr, in_1_ptr, in_0_ptr, out_ptr,
    B: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """Fused kernel for mul + add pattern."""
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    
    if b >= B:
        return
    
    # Compute offsets
    in_2_offset = b * H * D + h * D
    out_base = b * H * 2 * D + h * 2 * D
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < D
    
    # Load values
    in_2_vals = tl.load(in_2_ptr + in_2_offset + offsets, mask=mask, other=0.0)
    in_1_0 = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    in_1_1 = tl.load(in_1_ptr + D + offsets, mask=mask, other=0.0)
    in_0_0 = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    in_0_1 = tl.load(in_0_ptr + D + offsets, mask=mask, other=0.0)
    
    # Compute fused mul + add
    out0 = in_2_vals * in_1_0 + in_0_0
    out1 = in_2_vals * in_1_1 + in_0_1
    
    # Store in [B, H, 2, D] format for unbind(dim=2)
    tl.store(out_ptr + out_base + offsets, out0, mask=mask)
    tl.store(out_ptr + out_base + D + offsets, out1, mask=mask)


def triton_fused_impl(in_2, in_1, in_0):
    """Fused implementation matching mul+add pattern."""
    B, H, _, D = in_2.shape
    
    # Output as [B, H, 2, D] to be compatible with unbind(dim=2)
    out = torch.empty((B, H, 2, D), dtype=in_2.dtype, device=in_2.device)
    
    grid = (B * H,)
    triton_fused_kernel[grid](in_2, in_1, in_0, out, B, H, D)
    
    return out


@torch.fx.wrap
def triton_fused_impl_wrapped(in_2, in_1, in_0):
    return triton_fused_impl(in_2, in_1, in_0)


def pattern(in_0, in_1, in_2):
    """Match mul + add pattern for fusion."""
    tmp_0 = in_0
    tmp_1 = in_2 * in_1
    tmp_2 = tmp_1 + tmp_0
    return tmp_2


def replacement_args(in_0, in_1, in_2):
    """Return arguments in order expected by replacement function."""
    return (in_2, in_1, in_0)


def replacement_func():
    """Return the replacement function."""
    return triton_fused_impl_wrapped