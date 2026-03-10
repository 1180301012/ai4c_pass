import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    # Match the exact computation sequence from the original model:
    # tmp_0 = torch.nn.functional.softmax(in_1, dim=1)
    tmp_0 = torch.nn.functional.softmax(in_1, dim=1)
    
    # tmp_1 = tmp_0.reshape(32, -1) or (batch_size, -1)
    tmp_1 = tmp_0.reshape(tmp_0.shape[0], -1)
    
    # tmp_2 = tmp_1.view(batch_size, -1, 1, 1)
    tmp_2 = tmp_1.view(tmp_0.shape[0], -1, 1, 1)
    
    # tmp_3 = tmp_2.view(batch_size, 2, -1, 1, 1)
    tmp_3 = tmp_2.view(tmp_0.shape[0], 2, -1, 1, 1)
    
    # tmp_4 = tmp_3 * in_0
    tmp_4 = tmp_3 * in_0
    
    # tmp_5 = torch.sum(tmp_4, dim=1)
    tmp_5 = torch.sum(tmp_4, dim=1)
    
    # tmp_6 = tmp_5.contiguous()
    tmp_6 = tmp_5.contiguous()
    
    return (tmp_6,)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def optimized_kernel(softmax_ptr, in_0_ptr, out_ptr, 
                     batch_size, c, h, w, d,
                     BLOCK_SIZE: tl.constexpr):
    # Program IDs: batch_id and spatial flatten index (h*w*d)
    batch_id = tl.program_id(0)
    spatial_idx = tl.program_id(1)
    pid = spatial_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Calculate spatial dimensions index
    h_idx = pid // (w * d)
    w_idx = (pid % (w * d)) // d
    d_idx = pid % d
    mask = pid < h * w * d
    
    # Calculate linear indices for softmax_out shape [B, 2, H, 1, 1]
    # Each position (h, w, d) only accesses the value at w=0, d=0
    softmax_idx = batch_id * (2 * h * 1 * 1) + h_idx * (1 * 1)
    softmax_values = tl.load(softmax_ptr + softmax_idx.to(tl.int64), mask=pid < h * 1 * 1, other=0.0)
    
    # Calculate linear indices for in_0 shape [B, 2, H, W, D]
    in_0_idx = batch_id * (2 * h * w * d) + h_idx * (w * d) + w_idx * d + d_idx
    in_0_values = tl.load(in_0_ptr + in_0_idx.to(tl.int64), mask=mask)
    
    # Perform broadcasting multiplication: softmax_out[:, :, h, 0:1, 0:1] * in_0[:, :, h, w, d]
    # This effectively broadcasts [B, 2, H, 1, 1] x [B, 2, H, W, D] -> [B, 2, H, W, D]
    broadcast_mul = softmax_values * in_0_values
    
    # Sum along the channel dimension (dim=1) -> reduce from 2 channels to 1
    summed_result = tl.sum(broadcast_mul, axis=0)
    
    # Store result at [B, H, W, D]
    out_idx = batch_id * (h * w * d) + h_idx * (w * d) + w_idx * d + d_idx
    tl.store(out_ptr + out_idx.to(tl.int64), summed_result, mask=mask)

@torch.fx.wrap
def kernel_wrapper(in_0, in_1):
    # Get input shapes
    batch_size = in_1.shape[0]
    c = in_1.shape[1]  # Should be 2
    h = in_1.shape[2]  # Third dimension from softmax output
    w = in_0.shape[3]  # Second spatial dimension  
    d = in_0.shape[4]  # Third spatial dimension
    
    # Calculate total spatial dimensions
    spatial_size = h * w * d
    
    # Set block size and grid dimensions
    BLOCK_SIZE = 1024
    num_spatial_programs = (spatial_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (batch_size, num_spatial_programs)
    
    # Create output tensor
    out_shape = (batch_size, h, w, d)
    output = torch.empty(out_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Launch kernel
    optimized_kernel[grid](
        in_1,  # softmax output
        in_0,  # original input
        output,
        batch_size,
        c,
        h,
        w,
        d,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return kernel_wrapper