import torch
import triton
import triton.language as tl

# Pattern for post-conv operations variant 2: dropout -> scale -> add
# with different residual position
def pattern(conv_out, scale, residual):
    tmp_6 = torch.nn.functional.dropout(conv_out, 0.0, False, False)
    tmp_7 = scale.unsqueeze(-1)
    tmp_8 = tmp_7.unsqueeze(-1)
    tmp_9 = tmp_8 * tmp_6
    tmp_10 = residual + tmp_9
    return tmp_10

def replacement_args(conv_out, scale, residual):
    return (conv_out, scale, residual)


@triton.jit
def fused_scale_add_kernel_v2(
    conv_out_ptr,      # Conv output [N, C, H, W]
    residual_ptr,      # Residual [N, C, H, W]
    scale_ptr,         # Scale [C]
    out_ptr,           # Output [N, C, H, W]
    N, C, HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N * C * HW
    mask = offsets < total
    
    # Convert flat offset to (n, c, hw) for NCHW layout
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
def fused_scale_add_v2(conv_out, scale, residual):
    """
    Fused kernel for: dropout(noop) -> scale -> add
    """
    N, C, H, W = conv_out.shape
    HW = H * W
    total = N * C * HW
    
    out = residual.new_empty((N, C, H, W))
    
    BLOCK_SIZE = 1024
    grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    fused_scale_add_kernel_v2[grid](
        conv_out, residual, scale, out,
        N, C, HW, BLOCK_SIZE
    )
    
    return out


def replacement_func():
    return fused_scale_add_v2