import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """
    Match the pattern: scale * softmax + matmul + permute
    This is the core attention-like computation:
    1. tmp_0 = 0.0625 * in_0 (scale)
    2. tmp_1 = softmax(tmp_0, dim=-1)
    3. matmul = tmp_1 @ in_1
    4. tmp_3 = matmul.permute(0, 2, 1)
    
    Returns both matmul (intermediate) and permute result (final output).
    """
    tmp_0 = 0.0625 * in_0
    tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
    matmul = torch.matmul(tmp_1, in_1)
    tmp_3 = matmul.permute(0, 2, 1)
    return matmul, tmp_3


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_softmax_matmul_transpose_kernel(
    # Input pointers
    in_0_ptr, in_1_ptr,
    # Output pointer
    out_ptr,
    # Shapes
    batch_stride_in_0, seq_stride_in_0, feat_stride_in_0,
    batch_stride_in_1, feat_stride_in_1, out_stride_in_1,
    batch_stride_out, out_stride_out, seq_stride_out,
    # Dimensions
    batch_size, seq_len, feat_dim, out_dim,
    # Scale factor
    scale: tl.constexpr,
    # Block sizes
    BLOCK_SIZE_SEQ: tl.constexpr,
    BLOCK_SIZE_OUT: tl.constexpr,
):
    """
    Fused kernel: scale + softmax + matmul + transpose
    
    Computation:
    - Input: in_0 [B, S, F], in_1 [B, F, O]
    - tmp_0 = scale * in_0
    - tmp_1 = softmax(tmp_0, dim=-1)  [B, S, F]
    - matmul = tmp_1 @ in_1  [B, S, O]
    - output = matmul.permute(0, 2, 1)  [B, O, S]
    """
    # Get program ids
    batch_idx = tl.program_id(0)
    out_idx = tl.program_id(1)
    seq_idx = tl.program_id(2)
    
    # Compute pointers for this program
    in_0_base = batch_idx * batch_stride_in_0 + seq_idx * seq_stride_in_0
    in_1_base = batch_idx * batch_stride_in_1 + out_idx * feat_stride_in_1
    out_base = batch_idx * batch_stride_out + out_idx * out_stride_out + seq_idx * seq_stride_out
    
    # Load all features for in_0 at this (batch, seq) position
    # Compute softmax
    feat_acc = tl.zeros([1], dtype=tl.float32)
    max_val = tl.float32(-1e9)
    
    # Find max for numerically stable softmax
    for i in range(feat_dim):
        ptr = in_0_base + i * feat_stride_in_0
        val = tl.load(ptr).to(tl.float32)
        max_val = tl.max(max_val, val)
    
    # Compute exponentials and sum
    exp_sum = tl.float32(0.0)
    exp_vals = []
    for i in range(feat_dim):
        ptr = in_0_base + i * feat_stride_in_0
        val = tl.load(ptr).to(tl.float32)
        scaled_val = val * scale
        exp_val = tl.exp(scaled_val - max_val)
        exp_vals.append(exp_val)
        exp_sum = exp_sum + exp_val
    
    # Normalize and compute output
    output = tl.float32(0.0)
    for i in range(feat_dim):
        softmax_val = exp_vals[i] / exp_sum
        # Load from in_1: in_1[batch, i, out_idx]
        in_1_ptr = in_1_base + i * out_stride_in_1
        in_1_val = tl.load(in_1_ptr).to(tl.float32)
        output = output + softmax_val * in_1_val
    
    # Store result (transposed layout: [batch, out, seq])
    tl.store(out_base, output.to(tl.float16))


@torch.fx.wrap
def fused_softmax_matmul_transpose_wrapper(in_0, in_1):
    """
    Wrapper function to launch the fused kernel.
    Handles all shapes and types.
    """
    B, S, F = in_0.shape
    _, _, O = in_1.shape
    
    # Allocate output tensor with transposed layout [B, O, S]
    out = torch.empty((B, O, S), dtype=in_0.dtype, device=in_0.device)
    
    # Strides for in_0 [B, S, F]
    in_0_b_stride = in_0.stride(0)
    in_0_s_stride = in_0.stride(1)
    in_0_f_stride = in_0.stride(2)
    
    # Strides for in_1 [B, F, O]
    in_1_b_stride = in_1.stride(0)
    in_1_f_stride = in_1.stride(1)
    in_1_o_stride = in_1.stride(2)
    
    # Strides for out [B, O, S]
    out_b_stride = out.stride(0)
    out_o_stride = out.stride(1)
    out_s_stride = out.stride(2)
    
    # Scale factor
    scale = 0.0625
    
    # Launch kernel
    # Grid: [batch, out_dim, seq_len]
    grid = (B, O, S)
    
    # Use BLOCK_SIZE = 1 since we process individual elements
    BLOCK_SIZE_SEQ = 1
    BLOCK_SIZE_OUT = 1
    
    fused_softmax_matmul_transpose_kernel[grid](
        in_0, in_1, out,
        in_0_b_stride, in_0_s_stride, in_0_f_stride,
        in_1_b_stride, in_1_f_stride, in_1_o_stride,
        out_b_stride, out_o_stride, out_s_stride,
        B, S, F, O,
        scale,
        BLOCK_SIZE_SEQ, BLOCK_SIZE_OUT,
    )
    
    return out


def replacement_func():
    return fused_softmax_matmul_transpose_wrapper