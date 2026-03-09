import torch
import triton
import triton.language as tl

def pattern(in_7, tmp_0, tmp_1, tmp_3, tmp_2):
    tmp_7 = torch.nn.functional.batch_norm(in_7, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
    return tmp_7

def replacement_args(in_7, tmp_0, tmp_1, tmp_3, tmp_2):
    return (in_7, tmp_0, tmp_1, tmp_3, tmp_2)

@triton.jit
def batch_norm_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    num_features,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one feature
    pid = tl.program_id(0)
    
    # Compute range for this program
    feat_idx = pid
    mask = feat_idx < num_features
    
    if not mask:
        return
    
    # Load parameters for this feature
    running_mean = tl.load(running_mean_ptr + feat_idx, mask=mask, other=0.0)
    running_var = tl.load(running_var_ptr + feat_idx, mask=mask, other=0.0)
    weight_val = tl.load(weight_ptr + feat_idx, mask=mask, other=1.0)
    bias_val = tl.load(bias_ptr + feat_idx, mask=mask, other=0.0)
    
    # Compute normalization parameters
    inv_std = tl.rsqrt(running_var + eps)
    
    # Process each sample in the batch
    for i in range(0, batch_size, BLOCK_SIZE):
        # Load input chunk
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask_local = offsets < batch_size
        
        x_ptr_chunk = x_ptr + offsets * num_features + feat_idx
        x_chunk = tl.load(x_ptr_chunk, mask=mask_local, other=0.0)
        
        # Batch normalization: (x - running_mean) * inv_std * weight + bias
        normalized = (x_chunk - running_mean) * inv_std
        out_chunk = normalized * weight_val + bias_val
        
        # Store output
        out_ptr_chunk = out_ptr + offsets * num_features + feat_idx
        tl.store(out_ptr_chunk, out_chunk, mask=mask_local)

@torch.fx.wrap
def triton_batch_norm(x, running_mean, running_var, weight, bias):
    batch_size, num_features = x.shape
    eps = 1e-05
    
    BLOCK_SIZE = 1024
    
    # Calculate grid size (one program per feature)
    num_programs = num_features
    
    # Create output tensor
    out = torch.zeros_like(x)
    
    # Launch kernel
    batch_norm_kernel[(num_programs,)](
        x_ptr=x,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        batch_size=batch_size,
        num_features=num_features,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return triton_batch_norm