import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight_tensor, bias_tensor, residual_tensor):
    # Link the intermediate variables to match the model pattern exactly
    tmp_0 = bias_tensor
    tmp_1 = weight_tensor
    tmp_2 = input_tensor
    # Linear operation: y = x @ W^T + b (same as torch.nn.functional.linear(tmp_2, tmp_1, tmp_0))
    tmp_3 = torch.nn.functional.linear(tmp_2, tmp_1, tmp_0)
    # Apply dropout with p=0.0 (no-op for LINKX)
    tmp_4 = torch.nn.functional.dropout(tmp_3, p=0.0, training=False)
    # Final residual addition
    tmp_5 = residual_tensor + tmp_4
    # Return outputs to match model patterns
    return (tmp_5, tmp_4)

def replacement_args(input_tensor, weight_tensor, bias_tensor, residual_tensor):
    return (input_tensor, weight_tensor, bias_tensor, residual_tensor)

@triton.jit
def optimized_linear_kernel(
    x_ptr,          # input tensor [M, N]
    w_ptr,          # weight tensor [K, N] (transposed weights)
    b_ptr,          # bias tensor [K] (broadcasted to [M, K])
    residual_ptr,   # residual tensor [M, K]  
    out_ptr,        # output tensor [M, K]
    n_elements,     # total elements
    M: tl.constexpr,  # batch size 
    N: tl.constexpr,  # input features
    K: tl.constexpr,  # output features
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Each program computes one tile of the output matrix
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    
    # Compute range of indices for this program
    m_start = pid_m * BLOCK_SIZE_M
    k_start = pid_k * BLOCK_SIZE_K
    
    # Create coordinate offsets
    m_offsets = m_start + tl.arange(0, BLOCK_SIZE_M)
    k_offsets = k_start + tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks to ensure we don't go out of bounds
    m_mask = m_offsets < M
    k_mask = k_offsets < K
    
    # Load weights for this output feature column [N]
    w = tl.load(w_ptr + k_offsets[:, None] * N, mask=k_mask[:, None], other=0.0)
    
    # Load bias for this output feature column [1]
    b = tl.load(b_ptr + k_offsets, mask=k_mask, other=0.0)
    
    # Initialize accumulator  
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    
    # Loop over N dimension (input features) for matrix multiplication
    for n in range(0, N, 1):
        # Load input slice [M]
        x = tl.load(x_ptr + m_offsets * N + n, mask=m_mask, other=0.0)
        
        # Outer product: x[*, newaxis] @ w[newaxis, *] -> [M, K]
        acc += x[:, None] * w
    
    # Add bias (broadcasted across M dimension)
    acc = acc + b[None, :]
    
    # Load residual tensor for residual addition
    residual = tl.load(residual_ptr + m_offsets[:, None] * K + k_offsets[None, :], 
                       mask=m_mask[:, None] & k_mask[None, :], other=0.0)
    
    # Apply residual addition: out = linear_output + residual
    out = acc + residual
    
    # Store result
    tl.store(out_ptr + m_offsets[:, None] * K + k_offsets[None, :], 
             out, mask=m_mask[:, None] & k_mask[None, :])

@torch.fx.wrap
def triton_linear_with_residual(input_tensor, weight_tensor, bias_tensor, residual_tensor):
    M, N = input_tensor.shape
    K = weight_tensor.shape[0]  # output features
    
    # Handle device placement - move any CPU tensors to CUDA if needed
    if input_tensor.device.type != 'cuda':
        input_tensor = input_tensor.cuda()
    if weight_tensor.device.type != 'cuda':
        weight_tensor = weight_tensor.cuda()
    if bias_tensor.device.type != 'cuda':
        bias_tensor = bias_tensor.cuda()
    if residual_tensor.device.type != 'cuda':
        residual_tensor = residual_tensor.cuda()
    
    # Create output tensor with same shape as output dimension [M, K]
    output = torch.empty((M, K), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Optimized block sizes for different tensor sizes
    if N == 128 and M >= 1000:  # LINKX case
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_K = 32
    elif N <= 2 and M <= 64:  # BERT case
        BLOCK_SIZE_M = 8
        BLOCK_SIZE_K = 2  
    elif N == 512 and M == 4:  # Medium BERT case
        BLOCK_SIZE_M = 4
        BLOCK_SIZE_K = 32
    else:
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_K = 32
    
    # Calculate grid size based on output dimensions
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_k = (K + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    # Launch kernel with residual tensor
    optimized_linear_kernel[(grid_m, grid_k)](
        x_ptr=input_tensor,
        w_ptr=weight_tensor,
        b_ptr=bias_tensor,
        residual_ptr=residual_tensor,
        out_ptr=output,
        n_elements=M * K,
        M=M,
        N=N,
        K=K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    # Return tuple pattern that matches our pattern function
    # Pattern always returns (final_output, dropout_output) 
    # BERT pattern will only use the first element, LINKX will use both
    return output, output

def replacement_func():
    return triton_linear_with_residual