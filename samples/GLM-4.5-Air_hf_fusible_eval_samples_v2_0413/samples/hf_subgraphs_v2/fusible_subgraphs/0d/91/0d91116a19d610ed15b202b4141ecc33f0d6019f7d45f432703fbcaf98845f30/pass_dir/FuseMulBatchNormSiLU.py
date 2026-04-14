import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    # Element-wise multiplication
    tmp_4 = in_5 * in_4
    
    # Store intermediate references for batch norm 
    # (this matches the pattern where inputs are temporarily stored)
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_2
    tmp_3 = in_3
    
    # Batch normalization (using the intermediate variables as in original)
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
    
    # SiLU activation
    tmp_6 = torch.nn.functional.silu(tmp_5, inplace=True)
    
    return (tmp_6,)

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)

@triton.jit
def fused_mul_batchnorm_silu_kernel(
    input_ptr,
    sigmoid_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    num_features,
    eps: tl.constexpr,
    momentum: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensors
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    sigmoid_val = tl.load(sigmoid_ptr + offsets, mask=mask, other=0.0)
    
    # Element-wise multiplication: input * sigmoid
    mul_result = input_val * sigmoid_val
    
    # Load batch norm parameters from CPU to GPU via cache
    feature_idx = pid % num_features
    running_mean = tl.load(running_mean_ptr + feature_idx, other=0.0)
    running_var = tl.load(running_var_ptr + feature_idx, other=1.0)
    weight = tl.load(weight_ptr + feature_idx, other=1.0)
    bias = tl.load(bias_ptr + feature_idx, other=0.0)
    
    # Compute batch normalization
    # y = weight * (x - running_mean) / sqrt(running_var + eps) + bias
    denom = tl.sqrt(running_var + eps)
    batch_norm_result = weight * (mul_result - running_mean) / denom + bias
    
    # SiLU activation: x * sigmoid(x)
    silu_result = batch_norm_result * tl.sigmoid(batch_norm_result)
    
    # Store result
    tl.store(output_ptr + offsets, silu_result, mask=mask)

@torch.fx.wrap
def fused_mul_batchnorm_silu(in_0, in_1, in_2, in_3, in_4, in_5):
    # Ensure input tensors are on the same device as the main computation
    device = in_5.device
    
    # Move batch norm parameters to the device if they're not already there
    if in_0.device != device:
        in_0 = in_0.to(device)
    if in_1.device != device:
        in_1 = in_1.to(device)
    if in_2.device != device:
        in_2 = in_2.to(device)
    if in_3.device != device:
        in_3 = in_3.to(device)
    
    # Get tensor shapes and properties
    input_numel = in_5.numel()
    num_features = in_0.shape[0]  # in_0 is running_mean with shape [num_features]
    
    # Determine optimal block size
    BLOCK_SIZE = 1024
    
    # Calculate number of programs needed
    num_programs = (input_numel + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty_like(in_5)
    
    # Launch the fused kernel
    if num_features > 0 and num_programs > 0:
        fused_mul_batchnorm_silu_kernel[(num_programs,)](
            input_ptr=in_5,
            sigmoid_ptr=in_4,
            running_mean_ptr=in_0,
            running_var_ptr=in_1,
            weight_ptr=in_3,
            bias_ptr=in_2,
            output_ptr=output,
            n_elements=input_numel,
            num_features=num_features,
            eps=1e-05,
            momentum=0.1,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return output

def replacement_func():
    return fused_mul_batchnorm_silu