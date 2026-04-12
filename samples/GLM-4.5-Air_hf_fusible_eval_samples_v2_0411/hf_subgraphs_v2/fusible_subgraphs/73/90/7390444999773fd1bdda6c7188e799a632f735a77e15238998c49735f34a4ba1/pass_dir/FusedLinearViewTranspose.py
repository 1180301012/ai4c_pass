import torch
import triton
import triton.language as tl

def pattern(a, b, c):
    """
    Simple pattern that matches linear + view + transpose sequence
    """
    # Linear transformation
    linear_output = torch.nn.functional.linear(c, b, a)
    
    # View operation - match from models
    viewed_output = linear_output.view(1, -1, 2, 64)
    
    # Transpose operation
    transposed_output = viewed_output.transpose(1, 2)
    
    return transposed_output



def replacement_args(a, b, c):
    """
    Extract arguments for the fused kernel
    """
    return (a, b, c)

@triton.jit
def fused_linear_view_transpose_kernel(
    weight_ptr, bias_ptr, hidden_states_ptr, 
    output_ptr, 
    batch_size, seq_len, hidden_dim, num_heads, head_size,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    """
    Fused kernel that combines linear transformation, view, and transpose
    Each program handles one output element [batch_idx, head_idx, seq_idx, head_dim_idx]
    """
    pid = tl.program_id(0)
    
    # Calculate output indices for this program
    total_output_elements = batch_size * num_heads * seq_len * head_size
    if pid >= total_output_elements:
        return
    
    # Convert flat index to 4D coordinates
    head_dim_idx = pid % head_size
    seq_idx = (pid // head_size) % seq_len
    head_idx = (pid // (head_size * seq_len)) % num_heads
    batch_idx = pid // (head_size * seq_len * num_heads)
    
    # Initialize output element
    output_value = 0.0
    
    # Compute dot product of appropriate vectors
    for d in range(hidden_dim):
        # Weight matrix element for this head and dimension
        weight_idx = (batch_idx * num_heads + head_idx) * hidden_dim * head_size + d * head_size + head_dim_idx
        weight_val = tl.load(weight_ptr + weight_idx, mask=d < hidden_dim, other=0.0)
        
        # Bias element for this head and dimension  
        bias_idx = (batch_idx * num_heads + head_idx) * hidden_dim + d
        bias_val = tl.load(bias_ptr + bias_idx, mask=d < hidden_dim, other=0.0)
        
        # Hidden states element for this position and dimension
        hidden_states_idx = (batch_idx * seq_len + seq_idx) * hidden_dim + d
        hidden_states_val = tl.load(hidden_states_ptr + hidden_states_idx, mask=d < hidden_dim, other=0.0)
        
        # Accumulate dot product
        output_value += hidden_states_val * weight_val + bias_val
    
    # Store final result
    output_idx = (batch_idx * num_heads + head_idx) * seq_len * head_size + seq_idx * head_size + head_dim_idx
    tl.store(output_ptr + output_idx, output_value)

@torch.fx.wrap
def fused_linear_view_transpose(a, b, c):
    """
    Wrapped function that launches the fused kernel
    """
    # Infer parameters from input tensors
    batch_size = c.shape[0]
    seq_len = c.shape[1]
    hidden_dim = b.shape[0]  # weight is [hidden_dim, hidden_dim]
    head_size = 64
    num_heads = 2  # Fixed based on the pattern view(1, -1, 2, 64)
    
    # Initialize output tensor in the correct transposed layout
    output = torch.empty((batch_size, num_heads, seq_len, head_size), 
                        dtype=c.dtype, device=c.device)
    
    # Launch the kernel with one program per output element
    total_output_elements = batch_size * num_heads * seq_len * head_size
    # Use block size for better parallelism
    BLOCK_SIZE = 256
    grid_size = (total_output_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_linear_view_transpose_kernel[(grid_size,)](
        b, a, c,  # weight, bias, hidden_states
        output,
        batch_size, seq_len, hidden_dim, num_heads, head_size,
        BLOCK_SIZE, BLOCK_SIZE
    )
    
    return output

def replacement_func():
    """
    Return the fused kernel function
    """
    return fused_linear_view_transpose