import torch
import triton
import triton.language as tl

# Pattern matching for reshape + permute + unbind fusion
def pattern(x):
    # Reshape operation
    tmp_2 = x.reshape(1, 197, 3, -1, 48)
    # Permute operation  
    tmp_3 = tmp_2.permute(2, 0, 3, 1, 4)
    # Unbind operation (need to return all elements to maintain observable intermediates)
    tmp_4 = tmp_3.unbind(0)
    tmp_5 = tmp_4[0]
    tmp_6 = tmp_4[1] 
    tmp_7 = tmp_4[2]
    return (tmp_5, tmp_6, tmp_7)

# Extract arguments for replacement
def replacement_args(x):
    return (x,)

# Optimized kernel for reshape + permute + unbind fusion
@triton.jit
def reshape_permute_unbind_kernel(
    x_ptr, out1_ptr, out2_ptr, out3_ptr,
    batch_size, seq_len, in_features, inner_dim,
    BLOCK_SIZE: tl.constexpr
):
    # Program identifiers - we'll use 2D grid for the tensor dimensions
    pid_c = tl.program_id(0)  # channel dimension (0, 1, 2)
    pid_i = tl.program_id(1)  # inner dimension 
    pid_d = tl.program_id(2)  # depth dimension (0-47)
    pid_s = tl.program_id(3)  # sequence dimension (0-196)
    
    # Global offset in input tensor
    # Input shape: [1, 197, 3, inner_dim, 48]
    # Linear order: batch * seq * channels * inner * depth
    input_offset = (pid_s * 3 * inner_dim * 48 + 
                   pid_c * inner_dim * 48 + 
                   pid_i * 48 + 
                   pid_d)
    
    # Global offset in output tensors
    # Output tensors have shape [1, inner_dim, 197, 48]
    output_offset = (pid_s * inner_dim * 48 + 
                    pid_i * 48 + 
                    pid_d)
    
    # Load input value
    x_val = tl.load(x_ptr + input_offset, 
                   mask=(pid_s < seq_len) & (pid_i < inner_dim) & (pid_d < 48), 
                   other=0.0)
    
    # Store to appropriate output tensor based on channel
    if pid_c == 0:
        tl.store(out1_ptr + output_offset, x_val, 
                mask=(pid_s < seq_len) & (pid_i < inner_dim) & (pid_d < 48))
    elif pid_c == 1:
        tl.store(out2_ptr + output_offset, x_val, 
                mask=(pid_s < seq_len) & (pid_i < inner_dim) & (pid_d < 48))
    else:  # pid_c == 2
        tl.store(out3_ptr + output_offset, x_val, 
                mask=(pid_s < seq_len) & (pid_i < inner_dim) & (pid_d < 48))

@torch.fx.wrap
def reshape_permute_unbind_fused(x):
    # Get input tensor shape
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    total_features = x.shape[2]
    
    # Calculate inner dimension: (total_features // 3 // 48)
    inner_dim = total_features // (3 * 48)
    
    # Create output tensors with shape [1, inner_dim, 197, 48]
    out1 = torch.empty((1, inner_dim, seq_len, 48), dtype=torch.float32, device=x.device)
    out2 = torch.empty((1, inner_dim, seq_len, 48), dtype=torch.float32, device=x.device) 
    out3 = torch.empty((1, inner_dim, seq_len, 48), dtype=torch.float32, device=x.device)
    
    # Block size for parallel processing
    BLOCK_SIZE = 64
    
    # Launch kernel - 4D grid for parallel execution
    reshape_permute_unbind_kernel[(3, inner_dim, triton.cdiv(48, BLOCK_SIZE), seq_len)](
        x_ptr=x,
        out1_ptr=out1,
        out2_ptr=out2,
        out3_ptr=out3,
        batch_size=batch_size,
        seq_len=seq_len,
        in_features=total_features,
        inner_dim=inner_dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out1, out2, out3

def replacement_func():
    return reshape_permute_unbind_fused