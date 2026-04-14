import torch
import triton
import triton.language as tl
from triton.testing import do_bench


def pattern(input_tensor, weight_tensor, bias_tensor, add_tensor):
    # Linear + Add + ReLU pattern matching
    linear = torch.nn.functional.linear(input_tensor, weight_tensor, bias_tensor)
    added = add_tensor + linear
    relu_out = added.relu_()
    return relu_out


def replacement_args(input_tensor, weight_tensor, bias_tensor, add_tensor):
    return (input_tensor, weight_tensor, bias_tensor, add_tensor)


@triton.jit
def linear_add_relu_kernel(
    input_ptr, weight_ptr, bias_ptr, add_ptr, output_ptr,
    batch_size, in_features, out_features,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute block range this program is responsible for
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    
    # Initialize accumulator for this block
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16)
    
    # Matrix multiplication loop over K dimension
    for k in range(0, in_features, BLOCK_SIZE_K):
        # Load input block
        input_off = m_start + tl.arange(0, BLOCK_SIZE_M)
        k_off = k + tl.arange(0, BLOCK_SIZE_K)
        
        # Create masks for boundary checking
        input_mask = (input_off < batch_size)[:, None] & (k_off < in_features)[None, :]
        
        # Load input and weight blocks efficiently with consistent block sizes
        input_ptrs = input_ptr + input_off[:, None] * in_features + k_off[None, :]
        weight_ptrs = weight_ptr + n_start[:, None] * in_features + k_off[None, :]
        
        input_block = tl.load(input_ptrs, mask=input_mask, other=0.0)
        weight_block = tl.load(weight_ptrs, mask=input_mask, other=0.0)
        
        # Transpose weight_block for matrix multiplication: [BLOCK_SIZE_N, BLOCK_SIZE_K] -> [BLOCK_SIZE_K, BLOCK_SIZE_N]
        weight_block_transposed = tl.trans(weight_block)
        
        # Matrix multiplication: [BLOCK_SIZE_M, BLOCK_SIZE_K] @ [BLOCK_SIZE_K, BLOCK_SIZE_N] -> [BLOCK_SIZE_M, BLOCK_SIZE_N]
        accumulator += tl.dot(input_block.to(tl.float32), weight_block_transposed.to(tl.float32)).to(tl.float16)
    
    # Add bias if provided
    if bias_ptr is not None:
        bias_off = n_start + tl.arange(0, BLOCK_SIZE_N)
        bias_mask = bias_off < out_features
        bias_val = tl.load(bias_ptr + bias_off, mask=bias_mask, other=0.0)
        accumulator += bias_val[None, :]
    
    # Add the input tensor (element-wise)
    if add_ptr is not None:
        add_off_m = m_start + tl.arange(0, BLOCK_SIZE_M)
        add_off_n = n_start + tl.arange(0, BLOCK_SIZE_N)
        add_mask = (add_off_m < batch_size)[:, None] & (add_off_n < out_features)[None, :]
        add_ptrs = add_ptr + add_off_m[:, None] * out_features + add_off_n[None, :]
        add_block = tl.load(add_ptrs, mask=add_mask, other=0.0)
        accumulator += add_block
    
    # Apply ReLU activation
    accumulator = tl.maximum(accumulator, 0.0)
    
    # Store result with proper masking
    m_off = m_start + tl.arange(0, BLOCK_SIZE_M)
    n_off = n_start + tl.arange(0, BLOCK_SIZE_N)
    output_mask = (m_off < batch_size)[:, None] & (n_off < out_features)[None, :]
    output_ptrs = output_ptr + m_off[:, None] * out_features + n_off[None, :]
    tl.store(output_ptrs, accumulator, mask=output_mask)


@torch.fx.wrap
def fused_linear_add_relu(input_tensor, weight_tensor, bias_tensor, add_tensor):
    # Handle different tensor data types more gracefully
    if input_tensor.is_cuda and weight_tensor.is_cpu and bias_tensor.is_cpu and add_tensor.is_cuda:
        # Standard case: weights/bias on CPU, inputs/adds on GPU
        pass
    else:
        # Handle other cases by copying tensors to appropriate devices
        # For this specific problem, we'll copy weights to GPU since that's more efficient
        if not weight_tensor.is_cuda:
            weight_tensor = weight_tensor.to(input_tensor.device)
        if bias_tensor is not None and not bias_tensor.is_cuda:
            bias_tensor = bias_tensor.to(input_tensor.device)
        if not add_tensor.is_cuda:
            add_tensor = add_tensor.to(input_tensor.device)
    
    batch_size, in_features = input_tensor.shape
    out_features = weight_tensor.shape[0]
    
    assert weight_tensor.shape == (out_features, in_features)
    if bias_tensor is not None:
        assert bias_tensor.shape == (out_features,)
    assert add_tensor.shape == (batch_size, out_features)
    
    # Use working block sizes first
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    
    # Calculate grid dimension
    grid_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (out_features + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (grid_m, grid_n)
    
    # Create output tensor
    output = torch.empty((batch_size, out_features), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel
    linear_add_relu_kernel[grid](
        input_tensor,
        weight_tensor,
        bias_tensor if bias_tensor is not None else None,
        add_tensor,
        output,
        batch_size,
        in_features,
        out_features,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
    )
    
    return output


def replacement_func():
    return fused_linear_add_relu