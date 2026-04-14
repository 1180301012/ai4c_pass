import torch
import triton
import triton.language as tl

def pattern(scale_factor, input_tensor):
    # Element-wise multiplication with scale factor
    multiplied = input_tensor * scale_factor
    # Make result contiguous
    result = multiplied.contiguous()
    return (result,)

def replacement_args(scale_factor, input_tensor):
    return (scale_factor, input_tensor)

@triton.jit
def fused_mul_contiguous_kernel(
    scale_ptr,
    input_ptr,
    out_ptr,
    N, C_out, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    # Program identifiers
    pid = tl.program_id(0)
    num_programs = tl.cdiv(int(N * C_out * H * W), BLOCK_SIZE)
    
    # Get block offsets  
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (N * C_out * H * W)
    
    # Load scale factor (broadcasted to spatial dimensions)
    # Scale factor has shape [1, C_out, 1, 1], we need to broadcast to full spatial extent
    scale_c = tl.load(scale_ptr + (offsets // (H * W)) % C_out, mask=offsets < C_out, other=0.0)
    
    # Broadcast scale to all spatial positions for this channel
    scale_flat = tl.broadcast_to(scale_c, [BLOCK_SIZE])
    
    # Load input tensor
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Perform multiplication
    out = x * scale_flat
    
    # Store result directly (effectively making it contiguous)
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_multiply_contiguous(scale_factor, input_tensor):
    # Get input shapes
    N, C_in, H_in, W_in = input_tensor.shape
    scale_N, scale_C, scale_H, scale_W = scale_factor.shape
    
    # Input validation
    if scale_N != 1 or scale_H != 1 or scale_W != 1 or scale_C != C_in:
        # Fallback to original implementation if shapes don't match
        multiplied = input_tensor * scale_factor
        result = multiplied.contiguous()
        return (result,)
    
    # Allocate output tensor
    output = torch.empty_like(input_tensor)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    total_elements = N * C_in * H_in * W_in
    grid = (tl.cdiv(total_elements, BLOCK_SIZE),)
    
    fused_mul_contiguous_kernel[grid](
        scale_ptr=scale_factor,
        input_ptr=input_tensor,
        out_ptr=output,
        N=N, C_out=C_in, H=H_in, W=W_in,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return (output,)

def replacement_func():
    return fused_multiply_contiguous