import torch
import triton
import triton.language as tl

# Pattern matching function for layer normalization
def pattern(tmp_7, tmp_2, tmp_1):
    # tmp_8 = torch.nn.functional.layer_norm(tmp_7, (320,), tmp_2, tmp_1, 1e-05)
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (320,), tmp_2, tmp_1, 1e-05)
    return tmp_8

# Argument extraction function
def replacement_args(tmp_7, tmp_2, tmp_1):
    return (tmp_7, tmp_2, tmp_1)

# Optimized layer normalization kernel
@triton.jit
def layernorm_kernel(
    x_ptr,           # input tensor [batch, seq, features]
    weight_ptr,      # weight [features]
    bias_ptr,        # bias [features]
    out_ptr,         # output tensor [batch, seq, features]
    mean_ptr,        # mean for each batch,seq [batch, seq]
    var_ptr,         # variance for each batch,seq [batch, seq]
    batch_size,      # batch dimension size
    seq_len,         # sequence length
    features,        # features dimension (320)
    eps,             # epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    # Load slice information for each element in the block
    pid_batch = tl.program_id(0)
    pid_seq = tl.program_id(1)
    pid_feature = tl.program_id(2)
    
    # Calculate memory offsets
    x_offset = pid_batch * seq_len * features + pid_seq * features + pid_feature * BLOCK_SIZE
    weight_offset = pid_feature * BLOCK_SIZE
    bias_offset = pid_feature * BLOCK_SIZE
    out_offset = pid_batch * seq_len * features + pid_seq * features + pid_feature * BLOCK_SIZE
    mean_offset = pid_batch * seq_len + pid_seq
    var_offset = pid_batch * seq_len + pid_seq
    
    block_start = x_offset
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (pid_batch * seq_len * features + (pid_seq + 1) * features)
    
    # Load input data for this block
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + weight_offset)
    bias = tl.load(bias_ptr + bias_offset)
    
    # Compute mean and variance (already precomputed, but we'll do them in the kernel)
    # For performance, we'll compute mean and variance in a separate kernel
    # Here we assume they're precomputed and loaded from memory
    mean = tl.load(mean_ptr + mean_offset)
    var = tl.load(var_ptr + var_offset)
    
    # Compute normalization: (x - mean) / sqrt(var + eps) * weight + bias
    sqrt_var = tl.sqrt(var + eps)
    normalized = (x - mean) / sqrt_var
    result = normalized * weight + bias
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@triton.jit
def layernorm_mean_var_kernel(
    x_ptr,           # input tensor [batch, seq, features]
    mean_ptr,        # output mean [batch, seq]
    var_ptr,         # output variance [batch, seq]
    batch_size,      # batch dimension size  
    seq_len,         # sequence length
    features,        # features dimension (320)
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_seq = tl.program_id(1)
    
    # Calculate the memory offset for this batch,seq slice
    x_offset = pid_batch * seq_len * features + pid_seq * features
    mean_offset = pid_batch * seq_len + pid_seq
    var_offset = pid_batch * seq_len + pid_seq
    
    # Load the entire slice for this batch,seq
    offsets_slice = tl.arange(0, features)
    x_slice = tl.load(x_ptr + x_offset + offsets_slice)
    
    # Compute mean
    mean = tl.sum(x_slice) / features
    tl.store(mean_ptr + mean_offset, mean)
    
    # Compute variance
    x_centered = x_slice - mean
    var = tl.sum(x_centered * x_centered) / features
    tl.store(var_ptr + var_offset, var)

@torch.fx.wrap
def optimized_layernorm(x, weight, bias, eps=1e-05):
    batch_size, seq_len, features = x.shape
    
    # Allocate temporary storage for mean and variance
    mean = torch.empty((batch_size, seq_len), dtype=torch.float32, device=x.device)
    var = torch.empty((batch_size, seq_len), dtype=torch.float32, device=x.device)
    
    # First pass: compute mean and variance
    grid = (batch_size, seq_len, 1)  # No feature dimension needed for mean/var
    
    layernorm_mean_var_kernel[grid](
        x,
        mean,
        var,
        batch_size,
        seq_len,
        features,
        BLOCK_SIZE=features
    )
    
    # Second pass: apply layer normalization
    output = torch.empty_like(x)
    
    BLOCK_SIZE = 64  # Process 64 features at a time
    num_blocks = (features + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_norm = (batch_size, seq_len, num_blocks)
    
    layernorm_kernel[grid_norm](
        x,
        weight,
        bias,
        output,
        mean,
        var,
        batch_size,
        seq_len,
        features,
        eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

# Replacement function
def replacement_func():
    return optimized_layernorm