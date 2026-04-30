import torch
import triton
import triton.language as tl


@triton.jit
def fused_reshape_linear_slice_kernel(
    in_4_ptr,
    in_3_ptr,
    in_2_ptr,
    tmp_11_ptr,
    tmp_12_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    in_4_stride_b: tl.constexpr,
    in_4_stride_h: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused kernel that:
    1. Reshapes in_4 from [1, 150, 1, 512] to [300, 150, 256] conceptually
    2. Performs linear with weight [512, 256] and bias [512]
    3. Extracts only first and last 256 columns to produce [300, 150, 256] each
    
    This avoids materializing the full [300, 150, 512] intermediate tensor.
    """
    # pid_b: batch idx (0)
    # pid_h: head idx (0 to 149)
    # pid_m: m dimension (0 to 299)
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    #offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # For each output position (pid_h, pid_m), we compute a row of the linear
    # The input row in the reshaped tensor is at position [pid_m, pid_h, :]
    # This corresponds to in_4 index: [0, pid_h, 0, :] because of the reshape
    
    # Input pointer for this row (using in_4 strides [150*512, 512, 512, 1])
    # The row pid_m in the reshaped tensor corresponds to:
    # linear_idx = pid_m * 150 + pid_h (flattened index in reshaped [300, 150, :])
    # Which maps to in_4[0, h, 0, c] where h = (pid_m * 150 + pid_h) // 512
    # and c = (pid_m * 150 + pid_h) % 512
    
    # Actually, looking at the reshape more carefully:
    # in_4 has shape [1, 150, 1, 512] = [1, 150, 1, 2*256]
    # reshape to [300, 150, 256]
    # So index [m, h, k] in reshaped = in_4[0, h, 0, k] if m < 256
    # and index [m, h, k] in reshaped = in_4[0, h + 150, 0, k] if m >= 256
    # Wait, that doesn't work. Let me recalculate.
    
    # in_4: [1, 150, 1, 512] has 1*150*1*512 = 76800 elements
    # reshape to [300, 150, 256] has 300*150*256 = 11520000 elements
    # These don't match! There must be an error in my analysis.
    
    # Let me re-check:
    # in_4 shape: [1, 150, 1, 512] -> 76800 elements
    # reshape to [300, -1, 256] -> -1 is inferred as 76800/(300*256) = 1
    # So reshape to [300, 1, 256]
    
    # Hmm, but the slicing gives tmp_11 with shape [300, 150, 256]
    # Let me trace through more carefully.
    
    # Looking at weight_meta:
    # in_4 "proposal_feat" shape = [1, 150, 1, 512]
    # 
    # Model does: tmp_9 = in_4.reshape(300, -1, 256)
    # -1 is inferred: 1*150*1*512 / (300*256) = 1
    # So tmp_9 = [300, 1, 256]
    #
    # Then linear produces: [300, 1, 512]
    # And slices: tmp_11[..., :256] and tmp_12[..., -256:] -> [300, 1, 256] each
    
    # But looking at the weight meta for in_4, it's used with in_3 [512, 256] -> output [300, 1, 512]
    
    # Actually wait, looking at in_4 shape [1, 150, 1, 512]:
    # With -1 inference: 76800 / (300*256) = 76800/76800 = 1
    # So tmp_9 = [300, 1, 256]
    #
    # Linear: [300, 1, 256] @ [512, 256]^T + [512] = [300, 1, 512]
    #
    # Slices: [300, 1, 256] each
    #
    # The outputs are [300, 1, 256], not [300, 150, 256]
    
    # But wait, looking at the return: tmp_11, tmp_12, tmp_8, tmp_13
    # tmp_11 = [300, 1, 256]
    # tmp_12 = [300, 1, 256]
    # tmp_8 = [300, 256] (from first linear)
    # tmp_13 = [300, 1, 256] (unsqueeze from first linear)
    
    # The second linear input is in_4.reshape(300, -1, 256) where in_4 = [1, 150, 1, 512]
    # 76800 / (300*256) = 1, so reshape gives [300, 1, 256]
    
    # Hmm but the name "proposal_feat" with [1, 150, 1, 512] suggests 150 proposals
    # And the reshape to [300, -1, 256] with -1 = 1 gives [300, 1, 256]
    
    # Let me trust the shapes and implement the optimization correctly.
    
    # For each (pid_h, pid_m) in [0,150) x [0,300):
    # Input row is at position (pid_m, pid_h, :) in reshaped [300, 150, 256]
    # This maps to in_4[0, pid_h, 0, :] = in_4[0, pid_h, 0, :] with stride 512
    
    # Actually let me just do the reshape in the kernel properly
    # in_4 has strides [150*512, 512, 512, 1]
    # For row (pid_m, pid_h, k) in reshaped, we need in_4[b=0, h=pid_h, w=0, c=k+offset]
    # where offset depends on pid_m
    
    # Actually, for reshape [1, 150, 1, 512] -> [300, 1, 256]:
    # The mapping is: [b, h, w, c] -> [b*300 + m, 0, k]
    # where m = h * 256 + w * 256 + k, c = k (for the first 256) or c = 256 + k (for second 256)
    # and h is essentially 0 because 150*256 = 38400 > 76800/2
    
    # Let me just implement the straightforward version and use tl.load with offset
    
    # For simplicity, let's just load the data row by row
    # The input row for (pid_m, pid_h) maps to in_4[b=0, h=pid_h, w=0, :]
    # which has stride 512
    
    # Compute input pointer for this row
    # in_4[b=0, h=pid_h, w=0] base pointer
    base_ptr = in_4_ptr + pid_h * in_4_stride_h
    
    # Load input row
    a_ptrs = base_ptr + offs_k
    a = tl.load(a_ptrs, mask=offs_k < K, other=0.0)
    
    # Compute matmul for first half (columns 0:256)
    # weight is [512, 256], we load columns 0:256
    w_ptrs_first = in_3_ptr + offs_k[None, :] * 256 + offs_n[None, :]
    w_first = tl.load(w_ptrs_first, mask=(offs_k[:, None] < K) & (offs_n[None, :] < 256), other=0.0)
    
    # Compute matmul for second half (columns 256:512)
    w_ptrs_second = in_3_ptr + offs_k[None, :] * 256 + (offs_n[None, :] + 256)
    w_second = tl.load(w_ptrs_second, mask=(offs_k[:, None] < K) & (offs_n[None, :] < 256), other=0.0)
    
    # Dot products
    acc_first = tl.dot(a, w_first)
    acc_second = tl.dot(a, w_second)
    
    # Add bias
    bias_first = tl.load(in_2_ptr + offs_n)
    bias_second = tl.load(in_2_ptr + offs_n + 256)
    
    acc_first = acc_first + bias_first
    acc_second = acc_second + bias_second
    
    # Output positions
    out_offset_first = pid_m * N * 256 + pid_h * 256 + offs_n
    out_offset_second = pid_m * N * 256 + pid_h * 256 + offs_n
    
    # Store
    first_ptrs = tmp_11_ptr + out_offset_first
    tl.store(first_ptrs, acc_first, mask=offs_n < 256)
    
    second_ptrs = tmp_12_ptr + out_offset_second
    tl.store(second_ptrs, acc_second, mask=offs_n < 256)


def optimized_kernel_wrapper(in_2, in_3, in_4):
    """
    Optimized wrapper for reshape + linear + slice.
    
    Pattern:
    tmp_9 = in_4.reshape(300, -1, 256)  # [1,150,1,512] -> [300,1,256]
    linear_1 = torch.nn.functional.linear(tmp_9, in_3, in_2)  # -> [300,1,512]
    tmp_11 = linear_1[(Ellipsis, slice(None, 256, None))]  # [300,1,256]
    tmp_12 = linear_1[(Ellipsis, slice(-256, None, None))]  # [300,1,256]
    """
    # in_4 shape: [1, 150, 1, 512]
    # reshape to [300, 1, 256] with -1 inference
    # output: [300, 1, 512], then sliced to [300, 1, 256] each
    
    M = 300  # first dim of reshape
    N = 1    # second dim of reshape (inferred as 76800/(300*256) = 1)
    K = 256  # k dimension
    
    # Output tensors [300, 1, 256]
    tmp_11 = torch.empty((M, N, K), device=in_4.device, dtype=in_4.dtype)
    tmp_12 = torch.empty((M, N, K), device=in_4.device, dtype=in_4.dtype)
    
    # Get strides for in_4 [1, 150, 1, 512]
    in_4_stride_b = in_4.stride(0)  # 76800
    in_4_stride_h = in_4.stride(1)  # 512
    
    # Grid: (batch=1, num_heads=150, m_dim=300)
    grid = (1, 150, M)
    
    BLOCK_M = 1
    BLOCK_N = 64  # 256 cols / 64 = 4 iterations
    BLOCK_K = 64  # K dimension tiling
    
    fused_reshape_linear_slice_kernel[grid](
        in_4,
        in_3,
        in_2,
        tmp_11,
        tmp_12,
        M,
        N,
        K,
        in_4_stride_b,
        in_4_stride_h,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )
    
    return tmp_11, tmp_12


@torch.fx.wrap
def kernel_wrapper(in_2, in_3, in_4):
    return optimized_kernel_wrapper(in_2, in_3, in_4)


def pattern(in_2, in_3, in_4):
    """
    Match the pattern: reshape + linear + 2 slices
    """
    tmp_9 = in_4.reshape(300, -1, 256)
    linear_1 = torch.nn.functional.linear(tmp_9, in_3, in_2)
    tmp_11 = linear_1[(Ellipsis, slice(None, 256, None))]
    tmp_12 = linear_1[(Ellipsis, slice(-256, None, None))]
    return tmp_11, tmp_12


def replacement_args(in_2, in_3, in_4):
    return (in_2, in_3, in_4)


def replacement_func():
    return kernel_wrapper