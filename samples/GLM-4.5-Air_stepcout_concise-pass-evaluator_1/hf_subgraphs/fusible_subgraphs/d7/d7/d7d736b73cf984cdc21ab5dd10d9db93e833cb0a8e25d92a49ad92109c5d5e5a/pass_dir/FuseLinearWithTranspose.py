import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1, in_2):
    """
    Matches the computation pattern: linear followed by transpose
    tmp_2 = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = tmp_2.transpose(-1, -2)
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.nn.functional.linear(in_2, tmp_1, tmp_0)
    tmp_3 = tmp_2.transpose(-1, -2)
    return tmp_3

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Simple fused kernel for linear + transpose operations
@triton.jit
def linear_transpose_kernel(
    x_ptr,           # Input tensor [B, M, K]
    weight_ptr,      # Weight tensor [N, K] 
    bias_ptr,        # Bias tensor [N]
    out_ptr,         # Output tensor [B, N, M]
    batch_size: tl.constexpr,
    m: tl.constexpr,     # 768
    n: tl.constexpr,     # 196  
    k: tl.constexpr,     # 196
):
    # Program identifiers - simplified to 2D grid
    pid = tl.program_id(0)
    pid_b = pid // ((m + 127) // 128 * (n + 127) // 128)
    pid_m = (pid % ((m + 127) // 128 * (n + 127) // 128)) // ((n + 127) // 128)
    pid_n = (pid % ((m + 127) // 128 * (n + 127) // 128)) % ((n + 127) // 128)
    
    # Work item offsets
    m_offset = pid_m * 128
    n_offset = pid_n * 128
    b_offset = pid_b
    
    # Bounds checking - avoid chained boolean operators
    if b_offset >= batch_size:
        return
    if m_offset >= m:
        return
    if n_offset >= n:
        return
    
    # Initialize accumulator
    accumulator = 0.0
    
    # Vector for this output location
    val = tl.zeros((128,), dtype=tl.float32)
    
    # Initialize accumulation for output [b_offset, m_offset:m_offset+128, n_offset:n_offset+128]
    # val[i, j] = sum_k (input[b_offset, m_offset + i, k] * weight[n_offset + j, k])
    
    # Load bias vector [j] for j in [n_offset, n_offset+127]
    bias_vector = tl.load(bias_ptr + n_offset + tl.arange(0, 128), 
                         mask=(n_offset + tl.arange(0, 128)) < n, other=0.0).to(tl.float32)
    
    # Loop over k dimension for matrix multiplication
    for k_offset in range(k):
        # Load weight row [j, k_offset] for j in [n_offset, n_offset+127] - this is weight[n_offset: n_offset+128, k_offset]
        weight_row = tl.load(weight_ptr + (n_offset + tl.arange(0, 128)) * k + k_offset, 
                           mask=(n_offset + tl.arange(0, 128)) < n, other=0.0).to(tl.float32)
        
        # Load input column [b_offset, :, k_offset] at positions m_offset:m_offset+128
        # This loads input[b_offset, m_offset, k_offset], input[b_offset, m_offset+1, k_offset], ...
        input_col = tl.load(x_ptr + b_offset * m * k + (m_offset + tl.arange(0, 128)) * k + k_offset, 
                          mask=(m_offset + tl.arange(0, 128)) < m, other=0.0).to(tl.float32)
        
        # Vector dot product: val[i, j] += weight_row[j] * input_col[i]
        val += weight_row * input_col
    
    # Add bias after the loop (broadcast bias vector across all i)
    val += bias_vector
    
    # Store output in transposed position
    out_index = b_offset * n * m + n_offset * m + (m_offset + tl.arange(0, 128))
    mask = ((n_offset < n) & (m_offset + tl.arange(0, 128) < m)) & (b_offset < batch_size)
    tl.store(out_ptr + out_index, val, mask=mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def linear_with_transpose_fused(input_tensor, weight, bias):
    # Handle different input tensor shapes
    if input_tensor.dim() == 3:
        # Input is [B, M, K] format
        batch_size, m, k = input_tensor.shape
        input_reshaped = input_tensor
    elif input_tensor.dim() == 2:
        # Input is [M, K] format (add batch dimension)
        m, k = input_tensor.shape
        batch_size = 1
        input_reshaped = input_tensor.unsqueeze(0)
    else:
        # Should not happen for this model, but handle gracefully
        # Extract k from weight tensor shape
        k = weight.size(1)
        m = input_tensor.numel() // k if k > 0 else 1
        batch_size = 1
        input_reshaped = input_tensor.reshape(batch_size, m, k)
    
    n = weight.size(0)  # 196
    
    # Create output tensor in transposed format [B, N, M]
    output = torch.empty((batch_size, n, m), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate total number of threads needed
    num_m_blocks = (m + 127) // 128
    num_n_blocks = (n + 127) // 128
    total_threads = batch_size * num_m_blocks * num_n_blocks
    
    # Calculate grid dimensions
    grid = (total_threads,)
    
    # Launch kernel (simplified signature)
    linear_transpose_kernel[grid](
        input_reshaped,
        weight,
        bias,
        output,
        batch_size,
        m, n, k
    )
    
    # Remove batch dimension if it was added
    if input_tensor.dim() == 2:
        return output.squeeze(0)
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return linear_with_transpose_fused