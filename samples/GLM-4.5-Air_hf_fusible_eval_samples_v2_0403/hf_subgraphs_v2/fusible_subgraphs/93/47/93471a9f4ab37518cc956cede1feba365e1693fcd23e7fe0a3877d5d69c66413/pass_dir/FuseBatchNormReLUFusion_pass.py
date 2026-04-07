import torch
import triton
import triton.language as tl

def pattern(tmp_5, in_0, in_1, in_3, in_2, tmp_7):
    # Match batch_norm + relu pattern exactly as in graph
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = torch.nn.functional.relu(tmp_6, inplace=False)
    return tmp_7

def replacement_args(tmp_5, in_0, in_1, in_3, in_2, tmp_7):
    return (tmp_5, in_0, in_1, in_3, in_2, tmp_7)

@triton.jit
def fused_batch_norm_relu_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    N,           # batch size
    C,           # channels
    H,           # height  
    W,           # width
    eps: tl.constexpr,
    momentum: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Compute range for this program
    c_range = pid * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    n_range = tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks
    c_mask = c_range < C
    n_mask = n_range < N * H * W
    
    # Load input data
    input_ptrs = input_ptr + c_range[:, None] * H * W + n_range[None, :]
    input_data = tl.load(input_ptrs, mask=c_mask[:, None] & n_mask[None, :], other=0.0)
    
    # Load batch norm parameters
    running_mean_data = tl.load(running_mean_ptr + c_range, mask=c_mask, other=0.0).to(tl.float32)
    running_var_data = tl.load(running_var_ptr + c_range, mask=c_mask, other=0.0).to(tl.float32)
    weight_data = tl.load(weight_ptr + c_range, mask=c_mask, other=1.0).to(tl.float32)
    bias_data = tl.load(bias_ptr + c_range, mask=c_mask, other=0.0).to(tl.float32)
    
    # Apply batch normalization
    # Normalize input using running statistics
    input_normalized = (input_data.to(tl.float32) - running_mean_data[:, None]) / tl.sqrt(running_var_data[:, None] + eps)
    
    # Scale and shift
    batch_norm_output = input_normalized * weight_data[:, None] + bias_data[:, None]
    
    # Apply ReLU
    relu_output = tl.where(batch_norm_output > 0, batch_norm_output, 0.0)
    
    # Store result
    output_ptrs = output_ptr + c_range[:, None] * H * W + n_range[None, :]
    relu_output_cast = relu_output.to(input_data.dtype)
    tl.store(output_ptrs, relu_output_cast, mask=c_mask[:, None] & n_mask[None, :])

@torch.fx.wrap  
def fused_batch_norm_relu(tmp_5, in_0, in_1, in_3, in_2):
    # Determine output dtype based on input
    output_dtype = tmp_5.dtype
    
    batch_size, channels, height, width = tmp_5.shape
    
    output = torch.empty((batch_size, channels, height, width), 
                        dtype=output_dtype, device=tmp_5.device)
    
    # Launch kernel
    BLOCK_SIZE_N = 1024  # Number of elements per thread block
    BLOCK_SIZE_C = 256   # Number of channels per thread block
    
    grid = ( (channels + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C, )
    
    fused_batch_norm_relu_kernel[grid](
        input_ptr=tmp_5,
        running_mean_ptr=in_0,
        running_var_ptr=in_1, 
        weight_ptr=in_3,
        bias_ptr=in_2,
        output_ptr=output,
        N=batch_size * height * width,  # Flatten spatial dimensions
        C=channels,
        H=height,
        W=width,
        eps=1e-05,
        momentum=0.1,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )
    
    return output

def replacement_func():
    return fused_batch_norm_relu