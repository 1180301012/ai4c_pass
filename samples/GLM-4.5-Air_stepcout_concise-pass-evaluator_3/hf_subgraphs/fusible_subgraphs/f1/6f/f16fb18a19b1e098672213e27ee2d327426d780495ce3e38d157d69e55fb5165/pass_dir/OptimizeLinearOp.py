import torch
import triton
import triton.language as tl

# Pattern matching for just the linear operation
def pattern(in_0, in_1):
    """Match linear operation pattern"""
    tmp_1 = torch.nn.functional.linear(in_1, in_0, None)
    return tmp_1

def replacement_args(in_0, in_1):
    """Extract arguments for the optimized kernel"""
    return in_0, in_1

@triton.jit
def optimized_linear_kernel(
    weight_ptr,           # [D_out, D_in] weight matrix
    input_ptr,            # [batch, seq_len, D_in] input tensor  
    output_ptr,           # [batch, seq_len, D_out] output tensor
    batch,                # batch size (1)
    seq_len,              # sequence length (197)
    D_in,                 # input dimension (432/192)
    D_out,                # output dimension (1296/576)
    BLOCK_SIZE_K: tl.constexpr,  # block size for reduction
):
    """Optimized linear kernel using Triton"""
    pid_m = tl.program_id(0)  # sequence position
    pid_n = tl.program_id(1)  # output dimension position
    
    # Check bounds
    if pid_m >= seq_len or pid_n >= D_out:
        return
    
    # Initialize accumulator for this element
    acc = 0.0
    
    # Vectorized reduction over D_in dimension
    for k in range(0, D_in, BLOCK_SIZE_K):
        # Compute current k indices
        k_indices = k + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_indices < D_in
        
        # Load input and weights for current k indices
        input_vals = tl.load(input_ptr + pid_m * D_in + k_indices, mask=k_mask, other=0.0)
        weight_vals = tl.load(weight_ptr + pid_n * D_in + k_indices, mask=k_mask, other=0.0)
        
        # Accumulate dot product
        acc += tl.sum(input_vals * weight_vals)
    
    # Store result
    output_offset = pid_m * D_out + pid_n
    tl.store(output_ptr + output_offset, acc)

@torch.fx.wrap
def optimized_linear_forward(weight, input_tensor):
    """Forward pass using optimized linear kernel"""
    batch, seq_len, D_in = input_tensor.shape
    D_out = weight.shape[0]
    
    # Optimized linear operation using Triton kernel
    linear_output = torch.empty(batch, seq_len, D_out, dtype=torch.float32, device=input_tensor.device)
    
    # Launch optimized linear kernel - one thread per output element
    BLOCK_SIZE_K = 256  # Vectorization factor for D_in reduction
    
    grid_m = seq_len  # One thread per sequence position
    grid_n = D_out    # One thread per output dimension
    
    optimized_linear_kernel[(
        grid_m, 
        grid_n
    )](
        weight, 
        input_tensor, 
        linear_output,
        batch, 
        seq_len, 
        D_in, 
        D_out,
        BLOCK_SIZE_K
    )
    
    return linear_output

def replacement_func():
    return optimized_linear_forward