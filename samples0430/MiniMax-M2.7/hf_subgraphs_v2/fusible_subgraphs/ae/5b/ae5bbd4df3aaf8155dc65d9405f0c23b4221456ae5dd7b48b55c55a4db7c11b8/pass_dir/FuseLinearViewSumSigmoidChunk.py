import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    """
    Match the complete computation pattern:
    linear -> view -> sum -> sigmoid -> chunk -> mul -> sub -> mul -> add -> view
    
    The pattern returns tmp_14 (the final output).
    """
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_4 = linear.view(1, 12, 199, 2, 4)
    tmp_5 = tmp_4.sum(-1, keepdim=False)
    tmp_6 = torch.sigmoid(tmp_5)
    chunk = tmp_6.chunk(2, dim=-1)
    tmp_8 = chunk[0]
    tmp_9 = chunk[1]
    tmp_10 = tmp_9 * in_2
    tmp_11 = tmp_10 - 1.0
    tmp_12 = tmp_8 * tmp_11
    tmp_13 = tmp_12 + 2.0
    tmp_14 = tmp_13.view(1, 12, -1, 1)
    return tmp_14


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.jit
def fused_linear_view_sum_kernel(
    # Linear weights and bias
    weight_ptr, bias_ptr,
    # Input tensor
    input_ptr,
    # Output buffer for linear result
    linear_out_ptr,
    # Dimensions
    batch: tl.constexpr,
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    out_features: tl.constexpr,
    # Strides for input [batch, heads, seq, head_dim]
    input_batch_stride: tl.constexpr,
    input_head_stride: tl.constexpr,
    input_seq_stride: tl.constexpr,
    input_head_dim_stride: tl.constexpr,
    # Strides for weight [out_features, head_dim]
    weight_out_stride: tl.constexpr,
    # Strides for linear output [batch, heads, seq, out_features]
    linear_batch_stride: tl.constexpr,
    linear_head_stride: tl.constexpr,
    linear_seq_stride: tl.constexpr,
    linear_dim_stride: tl.constexpr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    First kernel: Compute linear layer output.
    linear = input @ weight.T + bias
    input: [batch, num_heads, seq_len, head_dim]
    weight: [out_features, head_dim]
    bias: [out_features]
    output: [batch, num_heads, seq_len, out_features]
    """
    # Get program IDs - each thread block handles one [batch, head, seq] position
    batch_pid = tl.program_id(0)
    head_pid = tl.program_id(1)
    seq_pid = tl.program_id(2)
    
    # Initialize accumulator
    acc = tl.zeros((out_features,), dtype=tl.float32)
    
    # Load bias
    offs_out = tl.arange(0, out_features)
    bias_mask = offs_out < out_features
    bias = tl.load(bias_ptr + offs_out, mask=bias_mask, other=0.0)
    acc = acc + tl.cast(bias, tl.float32)
    
    # Loop over head_dim for matmul
    offs_k = tl.arange(0, BLOCK_SIZE)
    for k in range(0, head_dim, BLOCK_SIZE):
        k_offs = k + offs_k
        k_mask = k_offs < head_dim
        
        # Load input: [batch, head, seq, k_offs]
        input_offset = (
            batch_pid * input_batch_stride +
            head_pid * input_head_stride +
            seq_pid * input_seq_stride +
            k_offs * input_head_dim_stride
        )
        input_vals = tl.load(input_ptr + input_offset, mask=k_mask, other=0.0)
        
        # Load weight: [outs, k_offs] for each output feature
        # weight[out, k] = weight_ptr[out * weight_out_stride + k]
        weight_offset = offs_out[:, None] * weight_out_stride + k_offs[None, :]
        weight_mask = (offs_out[:, None] < out_features) & (k_offs[None, :] < head_dim)
        weight_vals = tl.load(weight_ptr + weight_offset, mask=weight_mask, other=0.0)
        
        # Compute: input_vals * weight_vals, sum over k
        prod = input_vals[None, :] * weight_vals
        acc = acc + tl.sum(prod, axis=1)
    
    # Store linear output
    linear_offset = (
        batch_pid * linear_batch_stride +
        head_pid * linear_head_stride +
        seq_pid * linear_seq_stride
    )
    tl.store(linear_out_ptr + linear_offset + offs_out * linear_dim_stride, acc, mask=bias_mask)


@triton.jit
def fused_sigmoid_chunk_arith_kernel(
    # Linear output (viewed as [batch, heads, seq, 2, 4])
    linear_out_ptr,
    # Multiplicative constant
    mul_const_ptr,
    # Final output
    out_ptr,
    # Dimensions
    batch: tl.constexpr,
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    # Strides for linear output [batch, heads, seq, num_groups * group_size]
    linear_batch_stride: tl.constexpr,
    linear_head_stride: tl.constexpr,
    linear_seq_stride: tl.constexpr,
    linear_dim_stride: tl.constexpr,
    # Strides for output [batch, heads, seq, 1]
    output_batch_stride: tl.constexpr,
    output_head_stride: tl.constexpr,
    output_seq_stride: tl.constexpr,
):
    """
    Second kernel: 
    - View: [batch, heads, seq, 8] -> [batch, heads, seq, 2, 4] conceptually
    - Sum: reduce last dim (4 elements) -> [batch, heads, seq, 2]
    - Sigmoid
    - Chunk: split into 2 groups
    - Arithmetic: chunk[0] * (chunk[1] * mul_const - 1.0) + 2.0
    - View: reshape to [batch, heads, seq, 1]
    """
    # Get program IDs - each thread block handles one [batch, head, seq] position
    batch_pid = tl.program_id(0)
    head_pid = tl.program_id(1)
    seq_pid = tl.program_id(2)
    
    # Compute base offset for this position in linear output
    base_offset = (
        batch_pid * linear_batch_stride +
        head_pid * linear_head_stride +
        seq_pid * linear_seq_stride
    )
    
    # Load all 8 values for this position and sum in groups of 4
    # Group 0: indices 0,1,2,3; Group 1: indices 4,5,6,7
    v0 = tl.load(linear_out_ptr + base_offset + 0 * linear_dim_stride)
    v1 = tl.load(linear_out_ptr + base_offset + 1 * linear_dim_stride)
    v2 = tl.load(linear_out_ptr + base_offset + 2 * linear_dim_stride)
    v3 = tl.load(linear_out_ptr + base_offset + 3 * linear_dim_stride)
    v4 = tl.load(linear_out_ptr + base_offset + 4 * linear_dim_stride)
    v5 = tl.load(linear_out_ptr + base_offset + 5 * linear_dim_stride)
    v6 = tl.load(linear_out_ptr + base_offset + 6 * linear_dim_stride)
    v7 = tl.load(linear_out_ptr + base_offset + 7 * linear_dim_stride)
    
    sum0 = v0 + v1 + v2 + v3
    sum1 = v4 + v5 + v6 + v7
    
    # Apply sigmoid
    sigmoid0 = 1.0 / (1.0 + tl.exp(-sum0))
    sigmoid1 = 1.0 / (1.0 + tl.exp(-sum1))
    
    # Load multiplicative constant (broadcast from [1, num_heads, 1, 1])
    mul_const = tl.load(mul_const_ptr)
    
    # Compute: chunk0 * (chunk1 * mul_const - 1.0) + 2.0
    tmp11 = sigmoid1 * mul_const - 1.0
    result = sigmoid0 * tmp11 + 2.0
    
    # Store result
    out_offset = (
        batch_pid * output_batch_stride +
        head_pid * output_head_stride +
        seq_pid * output_seq_stride
    )
    tl.store(out_ptr + out_offset, result)


@torch.fx.wrap
def triton_linear_view_sum_sigmoid(in_0, in_1, in_2, in_3):
    """
    Fused kernel for the complete computation:
    linear -> view -> sum -> sigmoid -> chunk -> mul -> sub -> mul -> add -> view
    
    Args:
        in_0: bias, shape [8]
        in_1: weight, shape [8, 64]
        in_2: multiplicative constant, shape [1, num_heads, 1, 1]
        in_3: input tensor, shape [1, num_heads, 199, 64]
    
    Returns:
        output tensor, shape [1, num_heads, 199, 1]
    """
    # Get dimensions
    batch, num_heads, seq_len, head_dim = in_3.shape
    out_features = in_1.shape[0]  # 8
    
    # Allocate intermediate buffer for linear output: [batch, num_heads, seq_len, out_features]
    linear_out = torch.empty((batch, num_heads, seq_len, out_features), device=in_3.device, dtype=in_3.dtype)
    
    # Allocate output: [batch, num_heads, seq_len, 1]
    out = torch.empty((batch, num_heads, seq_len, 1), device=in_3.device, dtype=in_3.dtype)
    
    # Get strides
    input_batch_stride = in_3.stride(0)
    input_head_stride = in_3.stride(1)
    input_seq_stride = in_3.stride(2)
    input_head_dim_stride = in_3.stride(3)
    
    weight_out_stride = in_1.stride(0)
    
    linear_batch_stride = linear_out.stride(0)
    linear_head_stride = linear_out.stride(1)
    linear_seq_stride = linear_out.stride(2)
    linear_dim_stride = linear_out.stride(3)
    
    output_batch_stride = out.stride(0)
    output_head_stride = out.stride(1)
    output_seq_stride = out.stride(2)
    
    # Grid: (batch, num_heads, seq_len)
    grid = (batch, num_heads, seq_len)
    
    # Block size for matmul reduction
    BLOCK_SIZE = 64  # head_dim
    
    # Launch first kernel: linear layer
    fused_linear_view_sum_kernel[grid](
        in_1, in_0, in_3, linear_out,
        batch, num_heads, seq_len, head_dim, out_features,
        input_batch_stride, input_head_stride, input_seq_stride, input_head_dim_stride,
        weight_out_stride,
        linear_batch_stride, linear_head_stride, linear_seq_stride, linear_dim_stride,
        BLOCK_SIZE,
    )
    
    # Launch second kernel: sigmoid + chunk + arithmetic
    fused_sigmoid_chunk_arith_kernel[grid](
        linear_out, in_2, out,
        batch, num_heads, seq_len,
        linear_batch_stride, linear_head_stride, linear_seq_stride, linear_dim_stride,
        output_batch_stride, output_head_stride, output_seq_stride,
    )
    
    return out


def replacement_func():
    return triton_linear_view_sum_sigmoid