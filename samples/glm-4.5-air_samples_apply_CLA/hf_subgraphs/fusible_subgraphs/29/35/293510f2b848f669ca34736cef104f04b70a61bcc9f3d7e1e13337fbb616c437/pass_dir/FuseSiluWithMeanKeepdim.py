import torch
import triton
import triton.language as tl

# Pattern matching function - matches SILU followed by mean reduction with keepdim=True
def pattern(in_0):
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    tmp_1 = tmp_0.mean((2, 3), keepdim=True)
    return (tmp_0, tmp_1)


# Argument extraction function
def replacement_args(in_0):
    return (in_0,)


# Optimized fused kernel: SILU + Mean reduction with keepdim
@triton.jit
def silu_mean_keepdim_kernel(
    x_ptr,  # input tensor pointer
    out_silu_ptr,  # output SILU tensor pointer  
    out_mean_ptr,  # output mean tensor pointer
    n_elements,  # total elements in input tensor
    spatial_size,  # spatial dimension size (H * W)
    batch_size,  # batch_size * channels
    num_reduction_dims,  # number of dims being reduced (2 for H,W)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one spatial position in the tensor
    linear_idx = tl.program_id(0)
    h_idx = tl.program_id(1) 
    w_idx = tl.program_id(2)
    b_idx = linear_idx  # batch index (batch_size * channels)
    
    # Calculate global memory offset
    offset = b_idx * spatial_size + h_idx * (tl.shape(x_ptr, 3)) + w_idx
    
    # Load input element
    x = tl.load(x_ptr + offset, mask=offset < n_elements, other=0.0)
    
    # Compute SILU: x * sigmoid(x)
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    silu_val = x * sigmoid_x
    
    # Store SILU result
    tl.store(out_silu_ptr + offset, silu_val, mask=offset < n_elements)
    
    # For mean reduction, we store the mean value (will be divided by spatial_size later)
    tl.store(out_mean_ptr + offset, silu_val, mask=offset < n_elements)





@torch.fx.wrap
def fused_silu_mean_keepdim_forward(x):
    # Get tensor properties
    batch_size, channels, height, width = x.shape
    n_elements = x.numel()
    spatial_size = height * width
    
    # Create output tensors
    silu_out = torch.empty_like(x)
    mean_out = torch.empty(batch_size, channels, 1, 1, device=x.device)
    
    if spatial_size > 0:
        # Grid setup for spatial reduction
        batch_grid = batch_size * channels
        grid = (batch_grid, height, width)
        
        # Launch fused kernel
        silu_mean_keepdim_kernel[grid](
            x_ptr=x,
            out_silu_ptr=silu_out,
            out_mean_ptr=mean_out,
            n_elements=n_elements,
            spatial_size=spatial_size,
            batch_size=batch_size,
            num_reduction_dims=2,
            BLOCK_SIZE=1024
        )
        
        # Fix the mean values by dividing by spatial size
        mean_out = mean_out / (height * width)
    else:
        # Handle edge cases with empty spatial dimensions
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                silu_out[i, j, 0, 0] = x[i, j, 0, 0]  # Assuming single element
                mean_out[i, j, 0, 0] = x[i, j, 0, 0]
    
    return silu_out, mean_out


# Replacement function
def replacement_func():
    return fused_silu_mean_keepdim_forward