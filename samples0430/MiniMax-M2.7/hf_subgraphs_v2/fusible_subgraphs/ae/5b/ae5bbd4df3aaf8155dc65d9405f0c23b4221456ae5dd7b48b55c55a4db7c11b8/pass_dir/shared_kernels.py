"""
Shared Triton kernels for the fused linear + sigmoid optimization.
Both 12-head and 16-head passes use these kernels via the dispatch wrapper.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def fused_linear_kernel(
    weight_ptr, bias_ptr, input_ptr, linear_out_ptr,
    batch: tl.constexpr, num_heads: tl.constexpr, seq_len: tl.constexpr,
    head_dim: tl.constexpr, out_features: tl.constexpr,
    input_batch_stride: tl.constexpr, input_head_stride: tl.constexpr,
    input_seq_stride: tl.constexpr, input_head_dim_stride: tl.constexpr,
    weight_out_stride: tl.constexpr,
    linear_batch_stride: tl.constexpr, linear_head_stride: tl.constexpr,
    linear_seq_stride: tl.constexpr, linear_dim_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute linear layer: output = input @ weight.T + bias"""
    batch_pid = tl.program_id(0)
    head_pid = tl.program_id(1)
    seq_pid = tl.program_id(2)
    
    # Initialize accumulator in fp32 for precision
    acc = tl.zeros((out_features,), dtype=tl.float32)
    
    # Load bias and add
    offs_out = tl.arange(0, out_features)
    bias_mask = offs_out < out_features
    bias = tl.load(bias_ptr + offs_out, mask=bias_mask, other=0.0)
    acc = acc + bias.to(tl.float32)
    
    # Matmul loop: accumulate input @ weight.T
    offs_k = tl.arange(0, BLOCK_SIZE)
    for k in range(0, head_dim, BLOCK_SIZE):
        k_offs = k + offs_k
        k_mask = k_offs < head_dim
        
        # Load input tile
        input_offset = (
            batch_pid * input_batch_stride +
            head_pid * input_head_stride +
            seq_pid * input_seq_stride +
            k_offs * input_head_dim_stride
        )
        input_vals = tl.load(input_ptr + input_offset, mask=k_mask, other=0.0).to(tl.float32)
        
        # Load weight tile
        weight_offset = offs_out[:, None] * weight_out_stride + k_offs[None, :]
        weight_mask = (offs_out[:, None] < out_features) & (k_offs[None, :] < head_dim)
        weight_vals = tl.load(weight_ptr + weight_offset, mask=weight_mask, other=0.0).to(tl.float32)
        
        # Accumulate: [8] = sum over [64] of ([1,64] * [8,64])
        prod = input_vals[None, :] * weight_vals
        acc = acc + tl.sum(prod, axis=1)
    
    # Store result
    linear_offset = (
        batch_pid * linear_batch_stride +
        head_pid * linear_head_stride +
        seq_pid * linear_seq_stride
    )
    out_vals = acc.to(tl.float16)  # Match input dtype
    tl.store(linear_out_ptr + linear_offset + offs_out * linear_dim_stride, out_vals, mask=bias_mask)


@triton.jit
def fused_sigmoid_arith_kernel(
    linear_out_ptr, mul_const_ptr, out_ptr,
    batch: tl.constexpr, num_heads: tl.constexpr, seq_len: tl.constexpr,
    linear_batch_stride: tl.constexpr, linear_head_stride: tl.constexpr,
    linear_seq_stride: tl.constexpr, linear_dim_stride: tl.constexpr,
    output_batch_stride: tl.constexpr, output_head_stride: tl.constexpr,
    output_seq_stride: tl.constexpr,
):
    """Apply sigmoid, chunk, and arithmetic operations."""
    batch_pid = tl.program_id(0)
    head_pid = tl.program_id(1)
    seq_pid = tl.program_id(2)
    
    # Compute base offset
    base_offset = (
        batch_pid * linear_batch_stride +
        head_pid * linear_head_stride +
        seq_pid * linear_seq_stride
    )
    
    # Load 8 values and sum in groups of 4
    v0 = tl.load(linear_out_ptr + base_offset + 0 * linear_dim_stride).to(tl.float32)
    v1 = tl.load(linear_out_ptr + base_offset + 1 * linear_dim_stride).to(tl.float32)
    v2 = tl.load(linear_out_ptr + base_offset + 2 * linear_dim_stride).to(tl.float32)
    v3 = tl.load(linear_out_ptr + base_offset + 3 * linear_dim_stride).to(tl.float32)
    v4 = tl.load(linear_out_ptr + base_offset + 4 * linear_dim_stride).to(tl.float32)
    v5 = tl.load(linear_out_ptr + base_offset + 5 * linear_dim_stride).to(tl.float32)
    v6 = tl.load(linear_out_ptr + base_offset + 6 * linear_dim_stride).to(tl.float32)
    v7 = tl.load(linear_out_ptr + base_offset + 7 * linear_dim_stride).to(tl.float32)
    
    sum0 = v0 + v1 + v2 + v3
    sum1 = v4 + v5 + v6 + v7
    
    # Sigmoid in fp32 (tl.exp requires fp32/fp64)
    sigmoid0 = 1.0 / (1.0 + tl.exp(-sum0))
    sigmoid1 = 1.0 / (1.0 + tl.exp(-sum1))
    
    # Load multiplicative constant
    mul_const = tl.load(mul_const_ptr).to(tl.float32)
    
    # Arithmetic: sigmoid0 * (sigmoid1 * mul_const - 1.0) + 2.0
    tmp11 = sigmoid1 * mul_const - 1.0
    result = sigmoid0 * tmp11 + 2.0
    
    # Store result
    out_offset = (
        batch_pid * output_batch_stride +
        head_pid * output_head_stride +
        seq_pid * output_seq_stride
    )
    tl.store(out_ptr + out_offset, result.to(tl.float16))


@torch.fx.wrap
def triton_dispatch(in_0, in_1, in_2, in_3, num_heads, seq_len):
    """
    Dispatch wrapper that calls the appropriate Triton kernel.
    The num_heads parameter determines which kernel configuration to use.
    """
    batch, _, _, head_dim = in_3.shape
    out_features = in_1.shape[0]
    
    linear_out = torch.empty((batch, num_heads, seq_len, out_features), device=in_3.device, dtype=in_3.dtype)
    out = torch.empty((batch, num_heads, seq_len, 1), device=in_3.device, dtype=in_3.dtype)
    
    grid = (batch, num_heads, seq_len)
    BLOCK_SIZE = 64
    
    # First kernel: linear
    fused_linear_kernel[grid](
        in_1, in_0, in_3, linear_out,
        batch, num_heads, seq_len, head_dim, out_features,
        in_3.stride(0), in_3.stride(1), in_3.stride(2), in_3.stride(3),
        in_1.stride(0),
        linear_out.stride(0), linear_out.stride(1), linear_out.stride(2), linear_out.stride(3),
        BLOCK_SIZE,
    )
    
    # Second kernel: sigmoid + arithmetic
    fused_sigmoid_arith_kernel[grid](
        linear_out, in_2, out,
        batch, num_heads, seq_len,
        linear_out.stride(0), linear_out.stride(1), linear_out.stride(2), linear_out.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
    )
    
    return out