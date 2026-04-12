import torch
import triton
import triton.language as tl

@triton.jit
def fused_kernel(x1_ptr, x2_ptr, running_mean_ptr, running_var_ptr, 
                 weight_ptr, bias_ptr, out_bn_ptr, out_mean_ptr,
                 N, C, H, W, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_size = BLOCK_SIZE
    
    # Process spatial dimensions for each batch and channel
    for batch in range(N):
        for channel in range(C):
            # Compute mean across spatial dimensions for this batch and channel
            spatial_sum = 0.0
            
            # Process spatial block
            for h in range(H):
                for w in range(W):
                    x1 = tl.load(x1_ptr + batch * C * H * W + channel * H * W + h * W + w, other=0.0)
                    x2 = tl.load(x2_ptr + batch * C * H * W + channel * H * W + h * W + w, other=0.0)
                    spatial_sum += (x1 + x2)
            
            # Compute mean
            spatial_mean = spatial_sum / (H * W)
            
            # Apply batch normalization parameters
            running_mean = tl.load(running_mean_ptr + channel, other=0.0)
            running_var = tl.load(running_var_ptr + channel, other=1.0)
            weight_val = tl.load(weight_ptr + channel, other=1.0)
            bias_val = tl.load(bias_ptr + channel, other=0.0)
            
            # Batch norm formula: (x - mean) / sqrt(var + eps) * weight + bias
            eps = 1e-05
            normalized = (spatial_mean - running_mean) / tl.sqrt(running_var + eps)
            bn_output = normalized * weight_val + bias_val
            
            # Store results
            tl.store(out_bn_ptr + batch * C + channel, bn_output)
            tl.store(out_mean_ptr + batch * C + channel, spatial_mean)

@torch.fx.wrap
def fused_add_mean_batchnorm(x1, x2, running_mean, running_var, weight, bias):
    N, C, H, W = x1.shape
    C_mean = running_mean.shape[0]
    
    output_bn = torch.empty((N, C), dtype=x1.dtype, device=x1.device)
    output_mean = torch.empty((N, C), dtype=x1.dtype, device=x1.device)
    
    # Determine optimal block size
    total_elements = N * C
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_kernel[(num_programs,)](
        x1_ptr=x1,
        x2_ptr=x2,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_bn_ptr=output_bn,
        out_mean_ptr=output_mean,
        N=N, C=C_mean, H=H, W=W,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output_bn, output_mean

def pattern(x1, x2, running_mean, running_var, weight, bias):
    # Match the sequence: add -> mean -> batch_norm
    # (after dropout elimination)
    tmp_4 = x1 + x2
    tmp_5 = tmp_4.mean((2, 3), keepdim=False)
    tmp_8 = torch.nn.functional.batch_norm(tmp_5, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    return tmp_8, tmp_5

def replacement_args(x1, x2, running_mean, running_var, weight, bias):
    return (x1, x2, running_mean, running_var, weight, bias)

def replacement_func():
    return fused_add_mean_batchnorm