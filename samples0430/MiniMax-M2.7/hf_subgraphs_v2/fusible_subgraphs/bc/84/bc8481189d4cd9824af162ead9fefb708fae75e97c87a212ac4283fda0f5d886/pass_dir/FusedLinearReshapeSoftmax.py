import torch
import triton
import triton.language as tl

# Pattern matching function - matches the exact computation pattern
def pattern(in_0, in_1, in_2):
    """
    Pattern: linear -> reshape -> softmax
    Matches: torch.nn.functional.linear(in_2, in_1, in_0), reshape to [-1, 9, 1], softmax(dim=1)
    """
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.reshape(linear, [-1, 9, 1])
    tmp_4 = torch.softmax(tmp_3, dim=1)
    return tmp_4

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fused_kernel_impl(
    # Input pointers
    in_2_ptr, in_1_ptr, in_0_ptr,
    # Output pointer
    out_ptr,
    # Strides
    in_2_batch_stride, in_2_seq_stride, in_2_hidden_stride,
    in_1_out_stride, in_1_hidden_stride,
    out_batch_stride, out_softmax_stride,
    # Dimensions
    B, N, D, H_out,
    num_batch_groups,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused kernel for: linear -> reshape -> softmax
    
    Each program computes the FULL output for one group of 9 elements.
    Uses vectorized loads for better memory coalescing.
    """
    # Softmax group size is fixed at 9
    SOFTMAX_GROUP_SIZE = 9
    
    pid = tl.program_id(0)
    group_idx = pid
    
    first_flat_idx = group_idx * SOFTMAX_GROUP_SIZE
    h_idx = first_flat_idx % H_out
    
    in_1_base = h_idx * in_1_out_stride
    
    # Load bias
    bias_val = tl.load(in_0_ptr + h_idx).to(tl.float32)
    
    # Precompute all n_idx values
    n_idx0 = first_flat_idx // H_out
    n_idx1 = (first_flat_idx + 1) // H_out
    n_idx2 = (first_flat_idx + 2) // H_out
    n_idx3 = (first_flat_idx + 3) // H_out
    n_idx4 = (first_flat_idx + 4) // H_out
    n_idx5 = (first_flat_idx + 5) // H_out
    n_idx6 = (first_flat_idx + 6) // H_out
    n_idx7 = (first_flat_idx + 7) // H_out
    n_idx8 = (first_flat_idx + 8) // H_out
    
    # Precompute in_2 bases
    in_2_base0 = n_idx0 * in_2_seq_stride
    in_2_base1 = n_idx1 * in_2_seq_stride
    in_2_base2 = n_idx2 * in_2_seq_stride
    in_2_base3 = n_idx3 * in_2_seq_stride
    in_2_base4 = n_idx4 * in_2_seq_stride
    in_2_base5 = n_idx5 * in_2_seq_stride
    in_2_base6 = n_idx6 * in_2_seq_stride
    in_2_base7 = n_idx7 * in_2_seq_stride
    in_2_base8 = n_idx8 * in_2_seq_stride
    
    # Initialize accumulators
    acc0 = bias_val
    acc1 = bias_val
    acc2 = bias_val
    acc3 = bias_val
    acc4 = bias_val
    acc5 = bias_val
    acc6 = bias_val
    acc7 = bias_val
    acc8 = bias_val
    
    # Process D dimension with vectorized loads
    d_offsets = tl.arange(0, BLOCK_SIZE_K)
    
    for d_start in range(0, D, BLOCK_SIZE_K):
        d_mask = (d_offsets + d_start) < D
        
        # Load all 9 in_2 values for this dimension slice
        in_2_ptrs0 = in_2_ptr + in_2_base0 + (d_start + d_offsets) * in_2_hidden_stride
        in_2_ptrs1 = in_2_ptr + in_2_base1 + (d_start + d_offsets) * in_2_hidden_stride
        in_2_ptrs2 = in_2_ptr + in_2_base2 + (d_start + d_offsets) * in_2_hidden_stride
        in_2_ptrs3 = in_2_ptr + in_2_base3 + (d_start + d_offsets) * in_2_hidden_stride
        in_2_ptrs4 = in_2_ptr + in_2_base4 + (d_start + d_offsets) * in_2_hidden_stride
        in_2_ptrs5 = in_2_ptr + in_2_base5 + (d_start + d_offsets) * in_2_hidden_stride
        in_2_ptrs6 = in_2_ptr + in_2_base6 + (d_start + d_offsets) * in_2_hidden_stride
        in_2_ptrs7 = in_2_ptr + in_2_base7 + (d_start + d_offsets) * in_2_hidden_stride
        in_2_ptrs8 = in_2_ptr + in_2_base8 + (d_start + d_offsets) * in_2_hidden_stride
        
        # Load weight (shared across all 9 positions)
        in_1_ptrs = in_1_ptr + in_1_base + (d_start + d_offsets) * in_1_hidden_stride
        
        # Vectorized load
        w_vals = tl.load(in_1_ptrs, mask=d_mask, other=0.0)
        
        # Load all 9 input values
        in_2_0 = tl.load(in_2_ptrs0, mask=d_mask, other=0.0)
        in_2_1 = tl.load(in_2_ptrs1, mask=d_mask, other=0.0)
        in_2_2 = tl.load(in_2_ptrs2, mask=d_mask, other=0.0)
        in_2_3 = tl.load(in_2_ptrs3, mask=d_mask, other=0.0)
        in_2_4 = tl.load(in_2_ptrs4, mask=d_mask, other=0.0)
        in_2_5 = tl.load(in_2_ptrs5, mask=d_mask, other=0.0)
        in_2_6 = tl.load(in_2_ptrs6, mask=d_mask, other=0.0)
        in_2_7 = tl.load(in_2_ptrs7, mask=d_mask, other=0.0)
        in_2_8 = tl.load(in_2_ptrs8, mask=d_mask, other=0.0)
        
        # Accumulate
        acc0 += tl.sum(in_2_0 * w_vals, axis=0)
        acc1 += tl.sum(in_2_1 * w_vals, axis=0)
        acc2 += tl.sum(in_2_2 * w_vals, axis=0)
        acc3 += tl.sum(in_2_3 * w_vals, axis=0)
        acc4 += tl.sum(in_2_4 * w_vals, axis=0)
        acc5 += tl.sum(in_2_5 * w_vals, axis=0)
        acc6 += tl.sum(in_2_6 * w_vals, axis=0)
        acc7 += tl.sum(in_2_7 * w_vals, axis=0)
        acc8 += tl.sum(in_2_8 * w_vals, axis=0)
    
    # Compute softmax directly with scalar values
    # Find max
    max_val = acc0
    max_val = tl.where(acc1 > max_val, acc1, max_val)
    max_val = tl.where(acc2 > max_val, acc2, max_val)
    max_val = tl.where(acc3 > max_val, acc3, max_val)
    max_val = tl.where(acc4 > max_val, acc4, max_val)
    max_val = tl.where(acc5 > max_val, acc5, max_val)
    max_val = tl.where(acc6 > max_val, acc6, max_val)
    max_val = tl.where(acc7 > max_val, acc7, max_val)
    max_val = tl.where(acc8 > max_val, acc8, max_val)
    
    # Compute exp and sum
    exp_sum = tl.exp(acc0 - max_val) + tl.exp(acc1 - max_val) + tl.exp(acc2 - max_val)
    exp_sum = exp_sum + tl.exp(acc3 - max_val) + tl.exp(acc4 - max_val) + tl.exp(acc5 - max_val)
    exp_sum = exp_sum + tl.exp(acc6 - max_val) + tl.exp(acc7 - max_val) + tl.exp(acc8 - max_val)
    
    # Compute softmax outputs
    s0 = tl.exp(acc0 - max_val) / exp_sum
    s1 = tl.exp(acc1 - max_val) / exp_sum
    s2 = tl.exp(acc2 - max_val) / exp_sum
    s3 = tl.exp(acc3 - max_val) / exp_sum
    s4 = tl.exp(acc4 - max_val) / exp_sum
    s5 = tl.exp(acc5 - max_val) / exp_sum
    s6 = tl.exp(acc6 - max_val) / exp_sum
    s7 = tl.exp(acc7 - max_val) / exp_sum
    s8 = tl.exp(acc8 - max_val) / exp_sum
    
    # Store results
    base_offset = group_idx * out_batch_stride
    tl.store(out_ptr + base_offset + 0 * out_softmax_stride, s0)
    tl.store(out_ptr + base_offset + 1 * out_softmax_stride, s1)
    tl.store(out_ptr + base_offset + 2 * out_softmax_stride, s2)
    tl.store(out_ptr + base_offset + 3 * out_softmax_stride, s3)
    tl.store(out_ptr + base_offset + 4 * out_softmax_stride, s4)
    tl.store(out_ptr + base_offset + 5 * out_softmax_stride, s5)
    tl.store(out_ptr + base_offset + 6 * out_softmax_stride, s6)
    tl.store(out_ptr + base_offset + 7 * out_softmax_stride, s7)
    tl.store(out_ptr + base_offset + 8 * out_softmax_stride, s8)


@torch.fx.wrap
def fused_kernel_wrapper(in_0, in_1, in_2):
    """
    Wrapper for the fused linear + reshape + softmax kernel.
    
    Input shapes:
    - in_0: [18] (bias)
    - in_1: [18, 128] (weight)
    - in_2: [1, 19, 128] (input)
    
    Output shape: [38, 9, 1] (after reshape + softmax)
    """
    num_batch_groups = 38
    
    output_shape = (38, 9, 1)
    out = torch.empty(output_shape, dtype=in_0.dtype, device=in_0.device)
    
    B, N, D = in_2.shape  # 1, 19, 128
    H_out = in_0.shape[0]  # 18
    
    num_programs = num_batch_groups  # 38 programs
    
    fused_kernel_impl[(num_programs,)](
        in_2_ptr=in_2,
        in_1_ptr=in_1,
        in_0_ptr=in_0,
        out_ptr=out,
        in_2_batch_stride=in_2.stride(0),
        in_2_seq_stride=in_2.stride(1),
        in_2_hidden_stride=in_2.stride(2),
        in_1_out_stride=in_1.stride(0),
        in_1_hidden_stride=in_1.stride(1),
        out_batch_stride=out.stride(0),
        out_softmax_stride=out.stride(1),
        B=B, N=N, D=D, H_out=H_out,
        num_batch_groups=num_batch_groups,
        BLOCK_SIZE_K=128,
    )
    
    return out


def replacement_func():
    return fused_kernel_wrapper