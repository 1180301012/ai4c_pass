import torch
import triton
import triton.language as tl

# Pattern for post-conv operations: dropout -> scale -> add
# This matches: dropout(conv_out) * scale.unsqueeze(-1).unsqueeze(-1) + residual
def pattern(conv_out, scale, residual):
    tmp_6 = torch.nn.functional.dropout(conv_out, 0.0, False, False)
    tmp_7 = scale.unsqueeze(-1)
    tmp_8 = tmp_7.unsqueeze(-1)
    tmp_9 = tmp_8 * tmp_6
    tmp_10 = residual + tmp_9
    return tmp_10

def replacement_args(conv_out, scale, residual):
    return (conv_out, scale, residual)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['total'],
)
@triton.jit
def fused_scale_add_kernel(
    conv_out_ptr,      # Conv output [N, C, H, W]
    residual_ptr,      # Residual [N, C, H, W]
    scale_ptr,         # Scale [C]
    out_ptr,           # Output [N, C, H, W]
    C, HW, total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    
    # Convert flat offset to (n, c, hw) for NCHW layout
    # offset = n * C * HW + c * HW + hw
    remaining = offsets % (C * HW)
    c = remaining // HW
    
    # Load values
    conv_val = tl.load(conv_out_ptr + offsets, mask=mask, other=0.0)
    res_val = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    scale_val = tl.load(scale_ptr + c, mask=mask, other=0.0)
    
    # Compute output: scale * conv + residual
    out = scale_val * conv_val + res_val
    
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_scale_add(conv_out, scale, residual):
    """
    Fused kernel for: dropout(noop) -> scale -> add
    """
    N, C, H, W = conv_out.shape
    HW = H * W
    total = N * C * HW
    
    out = residual.new_empty((N, C, H, W))
    
    # Let autotuning choose the best block size
    grid = lambda meta: ((total + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    
    fused_scale_add_kernel[grid](
        conv_out, residual, scale, out,
        C, HW, total
    )
    
    return out


def replacement_func():
    return fused_scale_add