import torch
import triton
import triton.language as tl

# Pattern matching function for first linear layer
def pattern(x, weight, bias):
    """
    Pattern: torch.nn.functional.linear(x, weight, bias)
    This corresponds to: tmp_4 = torch.nn.functional.linear(in_5, tmp_1, tmp_0)
    - x: [300, 256] (in_5)
    - weight: [512, 256] (tmp_1/in_1) 
    - bias: [512] (tmp_0/in_0)
    - Output: [300, 512]
    """
    return torch.nn.functional.linear(x, weight, bias)

# Argument extraction function
def replacement_args(x, weight, bias):
    return (x, weight, bias)

# Triton kernel for optimized matrix multiplication with bias
@triton.jit
def linear_kernel_fp32(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    M,
    K,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Optimized linear layer: y = x @ weight.t() + bias
    M: batch size (300)
    K: input features (256) 
    N: output features (512)
    """
    # Program identifiers
    pid = tl.program_id(0)
    
    # Figure out rows this program should process
    m_block_start = pid * BLOCK_SIZE_M
    m_offsets = m_block_start + tl.arange(0, BLOCK_SIZE_M)
    m_mask = m_offsets < M
    
    # Each program handles a block of BLOCK_SIZE_M x N
    for k in range(0, K, BLOCK_SIZE_K):
        # Load block of weights
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < K
        
        weight_block = tl.load(
            weight_ptr + k_offsets[:, None] * N + tl.arange(0, BLOCK_SIZE_N)[None, :],
            mask=k_mask[:, None] & (tl.arange(0, BLOCK_SIZE_N)[None, :] < N),
            other=0.0
        )
        
        # Load block of x
        x_block = tl.load(
            x_ptr + m_offsets[:, None] * K + k_offsets[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0
        )
        
        # Matrix multiply part
        if 'acc' not in locals():
            acc = tl.dot(x_block, weight_block.to(tl.float32))
        else:
            acc += tl.dot(x_block, weight_block.to(tl.float32))
    
    # Load bias
    bias = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE_N), mask=tl.arange(0, BLOCK_SIZE_N) < N, other=0.0)
    
    # Add bias and store
    out = acc + bias
    tl.store(
        out_ptr + m_offsets[:, None] * N + tl.arange(0, BLOCK_SIZE_N)[None, :],
        out,
        mask=m_mask[:, None] & (tl.arange(0, BLOCK_SIZE_N)[None, :] < N)
    )

@triton.jit
def linear_kernel_basic(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    M,
    K,
    N,
):
    """
    Basic linear kernel processing one element at a time
    M: batch size (300)
    K: input features (256) 
    N: output features (512)
    """
    # Program identifiers - each program handles one output element
    batch_idx = tl.program_id(0)
    feat_idx = tl.program_id(1)
    
    # Check bounds
    if batch_idx >= M or feat_idx >= N:
        return
    
    # Load bias for this output element
    bias_val = tl.load(bias_ptr + feat_idx)
    acc = bias_val
    
    # Compute dot product: x[batch_idx, :] @ weight[feat_idx, :].t()
    for k in range(K):
        # Load input element
        x_val = tl.load(x_ptr + batch_idx * K + k)
        
        # Load weight element - weight has shape [N, K] = [512, 256]
        w_val = tl.load(weight_ptr + feat_idx * K + k)
        
        # Multiply and accumulate
        acc += x_val * w_val
    
    # Store result
    tl.store(out_ptr + batch_idx * N + feat_idx, acc)

@torch.fx.wrap
def optimized_linear(x, weight, bias):
    M, K = x.shape
    N = bias.shape[0]
    
    # For linear layer: input [M, K] @ weight.t() [K, N] + bias [N] = output [M, N]
    # So weight should have shape [N, K]
    assert weight.shape[0] == N, f"Weight first dim {weight.shape[0]} must match bias dim {N}"
    assert weight.shape[1] == K, f"Weight second dim {weight.shape[1]} must match input dim {K}"
    
    output_shape = (M, N)
    output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Calculate grid size - each program handles one output element
    grid = (M, N)
    
    # Launch the basic kernel
    linear_kernel_basic[grid](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=output,
        M=M,
        K=K,
        N=N
    )
    
    return output

# Replacement function
def replacement_func():
    return optimized_linear