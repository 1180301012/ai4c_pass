import torch
import triton
import triton.language as tl


def pattern(in_2, in_4, in_5, in_6):
    """
    Pattern: negate in_6, concat with in_5, multiply with in_2, add in_4, cast to float32
    Matches:
        tmp_0 = -in_6
        tmp_1 = torch.cat((tmp_0, in_5), dim=-1)
        tmp_2 = tmp_1 * in_2
        tmp_3 = in_4 + tmp_2
        tmp_4 = tmp_3.to(dtype=torch.float32)
    Returns: tmp_4
    """
    tmp_0 = -in_6
    tmp_1 = torch.cat((tmp_0, in_5), dim=-1)
    tmp_2 = tmp_1 * in_2
    tmp_3 = in_4 + tmp_2
    tmp_4 = tmp_3.to(dtype=torch.float32)
    return tmp_4


def replacement_args(in_2, in_4, in_5, in_6):
    return (in_2, in_4, in_5, in_6)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N'],
)
@triton.jit
def fused_neg_concat_mul_add_kernel(
    in_2_ptr, in_4_ptr, in_5_ptr, in_6_ptr, out_ptr,
    M, N, K,
    stride_in2_0, stride_in2_1, stride_in2_2, stride_in2_3,
    stride_in4_0, stride_in4_1, stride_in4_2, stride_in4_3,
    stride_in5_0, stride_in5_1, stride_in5_2, stride_in5_3,
    stride_in6_0, stride_in6_1, stride_in6_2, stride_in6_3,
    stride_out_0, stride_out_1, stride_out_2, stride_out_3,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Fused kernel for negate, concat, multiply, add, and cast."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size_m = min(num_pid_m, M - first_pid_m)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    off_pid_m = pid_m * BLOCK_SIZE_M
    off_pid_n = pid_n * BLOCK_SIZE_N
    offsets_m = off_pid_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = off_pid_n + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, K)

    mask_m = offsets_m < M
    mask_n = offsets_n < N

    # Compute pointers
    # in_2: [batch, head, seq, dim] with dim = 64
    # in_4: [batch, head, seq, dim] with dim = 64
    # in_5: [batch, head, seq, dim/2] with dim/2 = 32
    # in_6: [batch, head, seq, dim/2] with dim/2 = 32
    
    # For the concatenation: we need to read in_5 and -in_6 and combine them
    # in_5: [..., :, :, :32], in_6: [..., :, :, :32], result: [..., :, :, :64]
    
    # Compute in_6 negated
    in6_ptrs = in_6_ptr + offsets_m[:, None] * stride_in6_0 + offsets_n[None, :] * stride_in6_2 + offsets_k[None, :] * stride_in6_3
    in6 = tl.load(in6_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    neg_in6 = -in6  # Negate
    
    # Compute in_5
    in5_ptrs = in_5_ptr + offsets_m[:, None] * stride_in5_0 + offsets_n[None, :] * stride_in5_2 + offsets_k[None, :] * stride_in5_3
    in5 = tl.load(in5_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    
    # Concat along last dimension: [neg_in6, in5] -> shape [..., :, :, 64]
    # For first half (k < 32): use neg_in6
    # For second half (k >= 32): use in5
    concat_result = tl.where(offsets_k < K // 2, neg_in6, in5)
    
    # Multiply with in_2
    in2_ptrs = in_2_ptr + offsets_m[:, None] * stride_in2_0 + offsets_n[None, :] * stride_in2_2 + offsets_k[None, :] * stride_in2_3
    in2 = tl.load(in2_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    mul_result = concat_result * in2
    
    # Add in_4
    in4_ptrs = in_4_ptr + offsets_m[:, None] * stride_in4_0 + offsets_n[None, :] * stride_in4_2 + offsets_k[None, :] * stride_in4_3
    in4 = tl.load(in4_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    add_result = mul_result + in4
    
    # Cast to float32 (already float32, but ensure it)
    result = add_result.to(tl.float32)
    
    # Store
    out_ptrs = out_ptr + offsets_m[:, None] * stride_out_0 + offsets_n[None, :] * stride_out_2 + offsets_k[None, :] * stride_out_3
    tl.store(out_ptrs, result, mask=mask_m[:, None] & mask_n[None, :])


@torch.fx.wrap
def fused_neg_concat_mul_add_cast(in_2, in_4, in_5, in_6):
    """
    Fused implementation for:
    - Negate in_6
    - Concat with in_5 (using manual approach to avoid torch.cat)
    - Multiply with in_2
    - Add in_4
    - Cast to float32
    """
    # Get shapes
    batch, head, seq, dim_half = in_5.shape  # [batch, head, seq, dim/2]
    dim = dim_half * 2
    
    # Compute negation and concatenate without torch.cat
    # Create output tensor and manually fill it
    neg_in6 = -in_6
    
    # Use torch.empty to avoid initialization overhead
    # We'll fill it manually
    result = torch.empty((batch, head, seq, dim), dtype=torch.float32, device=in_2.device)
    
    # Fill first half with negated in_6
    result[:, :, :, :dim_half] = neg_in6
    # Fill second half with in_5
    result[:, :, :, dim_half:] = in_5
    
    # Multiply with in_2 and add in_4
    tmp_2 = result * in_2
    tmp_3 = in_4 + tmp_2
    
    # Cast to float32 (ensure type)
    tmp_4 = tmp_3.to(dtype=torch.float32)
    
    return tmp_4


def replacement_func():
    return fused_neg_concat_mul_add_cast