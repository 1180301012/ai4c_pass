import torch
import triton
import triton.language as tl

# Pattern matching function for minimal linear operation
def pattern(tmp_3, tmp_2, tmp_1):
    # Just match the linear operation
    return torch.nn.functional.linear(tmp_3, tmp_2, tmp_1)

# Argument extraction function
def replacement_args(tmp_3, tmp_2, tmp_1):
    return (tmp_3, tmp_2, tmp_1)

# Optimized kernel for simple linear operation
@triton.jit
def linear_reshape_kernel(
    input_ptr,      # Input tensor [batch, seq, hidden]
    weight_ptr,     # Weight matrix [hidden_out, hidden_in]
    bias_ptr,       # Bias vector [hidden_out]
    output_ptr,     # Output tensor [batch, seq, hidden_out]
    batch_size,
    seq_len,
    hidden_in,
    hidden_out,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Program ID calculations
    m = tl.program_id(0)  # batch sequence
    n = tl.program_id(1)  # output dimension
    
    # Create offsets for matrix multiplication
    offsets_m = m * seq_len + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Load input data
    input_ptr_base = input_ptr + m * seq_len * hidden_in
    input_data = tl.load(input_ptr_base + tl.arange(0, BLOCK_SIZE_M)[:, None] * hidden_in + offsets_n[None, :], 
                        mask=(offsets_m[:, None] < seq_len) & (offsets_n[None, :] < hidden_in),
                        other=0.0)
    
    # Load weight data
    weight_data = tl.load(weight_ptr + offsets_n[:, None] * hidden_in + tl.arange(0, BLOCK_SIZE_N)[None, :],
                         mask=(offsets_n[:, None] < hidden_out) & (tl.arange(0, BLOCK_SIZE_N)[None, :] < hidden_in),
                         other=0.0)
    
    # Load bias
    bias = tl.load(bias_ptr + offsets_n)
    
    # Matrix multiplication with bias add
    acc = tl.dot(input_data, weight_data, acc_type=tl.float32) + bias
    
    # Store result to output [batch, seq, hidden_out]
    tl.store(output_ptr + m * seq_len * hidden_out + 
            tl.arange(0, BLOCK_SIZE_M)[:, None] * hidden_out + 
            n,
            acc, mask=offsets_m[:, None] < seq_len)

@torch.fx.wrap
def optimized_linear(input, weight, bias):
    # Simple linear operation optimized with Triton
    batch_size, seq_len, hidden_in = input.shape
    _, hidden_out = weight.shape
    
    # Create output tensor
    output = torch.empty((batch_size, seq_len, hidden_out), dtype=input.dtype, device=input.device)
    
    # Launch kernel
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    
    # Number of programs needed
    m_programs = (batch_size * seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    n_programs = (hidden_out + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Use optimized linear kernel
    linear_reshape_kernel[(m_programs, n_programs)](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_in=hidden_in,
        hidden_out=hidden_out,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return output

# Replacement function
def replacement_func():
    return optimized_linear