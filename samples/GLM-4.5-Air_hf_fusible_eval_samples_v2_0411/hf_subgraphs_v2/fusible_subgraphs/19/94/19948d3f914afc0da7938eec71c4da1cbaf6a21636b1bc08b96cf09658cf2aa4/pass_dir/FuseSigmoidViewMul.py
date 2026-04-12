import torch
import triton
import triton.language as tl

def pattern(conv_out, scale_tensor):
    tmp_3 = torch.sigmoid(conv_out)
    tmp_4 = tmp_3.view(1, -1, 1, 1) 
    tmp_5 = scale_tensor * tmp_4
    return tmp_5

def replacement_args(conv_out, scale_tensor):
    return (conv_out, scale_tensor)

@triton.jit
def fused_sigmoid_view_mul_kernel(
    conv_out_ptr,
    scale_tensor_ptr, 
    out_ptr,
    n_channels,
    h,
    w,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    # Each program handles a channel block
    c_offset = tl.program_id(0) * BLOCK_SIZE_C
    hw_offset = tl.program_id(1) * BLOCK_SIZE_HW
    c_indices = c_offset + tl.arange(0, BLOCK_SIZE_C)
    hw_indices = hw_offset + tl.arange(0, BLOCK_SIZE_HW)
    
    # Mask for channel dimension
    c_mask = c_indices < n_channels
    # Mask for spatial dimensions  
    hw_mask = hw_indices < h * w
    
    # Load conv_out (shape: [1, n_channels, 1, 1])
    conv_out_val = tl.load(conv_out_ptr + c_indices, mask=c_mask, other=0.0)
    
    # Compute sigmoid and reshape to (1, n_channels, 1, 1) -> effectively broadcasting
    sigmoid_val = tl.sigmoid(conv_out_val.to(tl.float32)).to(conv_out_val.dtype)
    
    # Load scale_tensor spatial elements (shape: [1, n_channels, h, w])
    # hw_indices contains flattened spatial indices, create [BLOCK_SIZE_HW, BLOCK_SIZE_C] indices
    scale_indices = hw_indices[:, None] * n_channels + c_indices[None, :]
    scale_vals = tl.load(scale_tensor_ptr + scale_indices, mask=(hw_indices[:, None] < h * w) & c_mask[None, :], other=0.0)
    
    # Apply channel-wise scaling: scale_vals * sigmoid_val (broadcasting)
    out_vals = scale_vals * sigmoid_val
    
    # Store result
    out_indices = hw_indices[:, None] * n_channels + c_indices[None, :]
    tl.store(out_ptr + out_indices, out_vals, mask=(hw_indices[:, None] < h * w) & c_mask[None, :])

@torch.fx.wrap  
def fused_sigmoid_view_mul(conv_out, scale_tensor):
    n_channels = conv_out.shape[1]
    h, w = scale_tensor.shape[2], scale_tensor.shape[3]
    
    # Use smaller block sizes for better performance with these tensor shapes
    BLOCK_SIZE_C = 16   # Smaller channel block for better occupancy
    BLOCK_SIZE_HW = 64   # Moderate spatial block size
    
    # Calculate grid size
    grid_c = (n_channels + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    grid_hw = (h * w + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW
    grid_size = (grid_c, grid_hw)
    
    # Allocate output tensor
    out = torch.empty_like(scale_tensor)
    
    # Launch kernel
    fused_sigmoid_view_mul_kernel[grid_size](
        conv_out_ptr=conv_out,
        scale_tensor_ptr=scale_tensor,
        out_ptr=out,
        n_channels=n_channels,
        h=h, 
        w=w,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_HW=BLOCK_SIZE_HW,
    )
    
    return out

def replacement_func():
    return fused_sigmoid_view_mul