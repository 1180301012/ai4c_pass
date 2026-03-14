import torch
import triton
import triton.language as tl

# Pattern to match: dropout(p=0) + mul + add
# In the model: tmp_8 = dropout(tmp_7, 0.0, False, False), then tmp_9 = tmp_8 * scale, tmp_10 = residual + tmp_9
def pattern(x, scale, residual):
    dropped = torch.nn.functional.dropout(x, 0.0, False, False)
    scaled = dropped * scale
    added = residual + scaled
    return added

def replacement_args(x, scale, residual):
    return (x, scale, residual)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
    ],
    key=['total_elements'],
)
@triton.jit
def fused_dropout_mul_add_kernel(
    x_ptr,
    scale_ptr,
    residual_ptr,
    output_ptr,
    C, HW, C_HW, total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # For NCHW layout, compute channel index for each element
    c_indices = (offsets % C_HW) // HW
    
    # Load scale for each channel
    scale = tl.load(scale_ptr + c_indices, mask=mask)
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask)
    residual = tl.load(residual_ptr + offsets, mask=mask)
    
    # Compute: added = residual + x * scale (dropout is identity since p=0)
    added = residual + x * scale
    
    # Store result
    tl.store(output_ptr + offsets, added, mask=mask)

@torch.fx.wrap
def fused_dropout_mul_add(x, scale, residual):
    N, C, H, W = x.shape
    HW = H * W
    C_HW = C * HW
    total_elements = N * C_HW
    
    output = torch.empty_like(residual)
    
    # Squeeze scale from [C, 1, 1] to [C] for efficient loading
    scale_flat = scale.view(-1)
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    residual = residual.contiguous()
    
    grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
    
    fused_dropout_mul_add_kernel[grid](
        x,
        scale_flat,
        residual,
        output,
        C, HW, C_HW, total_elements,
    )
    
    return output

def replacement_func():
    return fused_dropout_mul_add