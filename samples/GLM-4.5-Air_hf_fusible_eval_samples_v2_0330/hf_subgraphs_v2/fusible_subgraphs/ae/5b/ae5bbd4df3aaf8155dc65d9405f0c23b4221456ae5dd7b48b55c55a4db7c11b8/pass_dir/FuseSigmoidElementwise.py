import torch
import triton
import triton.language as tl

def pattern(tmp_5, in_2):
    tmp_6 = torch.sigmoid(tmp_5)
    chunk = tmp_6.chunk(2, dim=-1)
    tmp_8 = chunk[0]
    tmp_9 = chunk[1]
    tmp_10 = tmp_9 * in_2
    tmp_11 = tmp_10 - 1.0
    tmp_12 = tmp_8 * tmp_11
    tmp_13 = tmp_12 + 2.0
    return tmp_13

def replacement_args(tmp_5, in_2):
    return (tmp_5, in_2)

@triton.jit
def sigmoid_elementwise_kernel(
    input_ptr, const_ptr, output_ptr,
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    features: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Output indices: [1, num_heads, seq_len, 1] (after chunk operation)
    output_idx = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask = output_idx < batch_size * num_heads * seq_len
    
    # Reorder indices for memory access
    output_idx_flat = output_idx[mask]
    m_idx = output_idx_flat
    
    # Load input for both parts (before chunking)
    # Input shape is [1, num_heads, seq_len, 2], we need both halves
    input_full = tl.load(input_ptr + m_idx * 2, mask=mask)
    input_first = input_full[0]  # First chunk element
    input_second = input_full[1]  # Second chunk element
    
    # Load constant
    const = tl.load(const_ptr).to(tl.float32)
    
    # Compute fused operations:
    # 1. Sigmoid on both inputs
    sigmoid_first = 1.0 / (1.0 + tl.exp(-input_first.to(tl.float32)))
    sigmoid_second = 1.0 / (1.0 + tl.exp(-input_second.to(tl.float32)))
    
    # 2. Multiply second element by constant
    multiplied = sigmoid_second * const
    
    # 3. Subtract 1.0
    subtracted = multiplied - 1.0
    
    # 4. Multiply first element by result
    result = sigmoid_first * subtracted
    
    # 5. Add 2.0
    final_result = result + 2.0
    
    # Store output (only first chunk element after operations)
    output_offset = output_idx_flat
    tl.store(output_ptr + output_offset, final_result.to(tl.float16), mask=mask)

@torch.fx.wrap
def sigmoid_elementwise_optimized(tmp_5, in_2):
    # Get input shapes
    batch_size, num_heads, seq_len, features = tmp_5.shape
    
    # Calculate output shape after operations: [1, num_heads, seq_len, 1]
    output_shape = [1, num_heads, seq_len, 1]
    
    # Create output tensor
    output = torch.empty(output_shape, dtype=torch.float16, device=tmp_5.device)
    
    # Launch kernel
    BLOCK_SIZE_M = 256
    
    grid = (batch_size * num_heads * seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
    
    sigmoid_elementwise_kernel[grid](
        input_ptr=tmp_5,
        const_ptr=in_2,
        output_ptr=output,
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        features=features,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
    )
    
    return output

def replacement_func():
    return sigmoid_elementwise_optimized