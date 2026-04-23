import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.flatten(conv2d, 2)
    return (tmp_3,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def conv1x1_flatten_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_batch_b, stride_batch_c,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Batched matmul + bias: C[batch, M, N] = A[M, K] @ B[batch, K, N] + bias[M]
    # A = weight [C_out, C_in], B = input [N_batch, C_in, HW], C = output [N_batch, C_out, HW]
    # A is shared across batches
    
    pid = tl.program_id(0)
    
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_per_batch = num_pid_m * num_pid_n
    
    batch_id = pid // num_pid_per_batch
    pid_in_batch = pid % num_pid_per_batch
    
    # Super-group for L2 locality
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid_in_batch // num_pid_in_group
    first_pid_m_in_group = group_id * GROUP_M
    group_size = min(GROUP_M, num_pid_m - first_pid_m_in_group)
    
    pid_m_in_group = pid_in_batch % num_pid_in_group
    pid_m = first_pid_m_in_group + (pid_m_in_group % group_size)
    pid_n = (pid_m_in_group // group_size)
    
    # Offsets
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    m_mask = m_offsets < M
    n_mask = n_offsets < N
    
    # Load bias
    bias = tl.load(bias_ptr + m_offsets, mask=m_mask, other=0.0)
    
    # Accumulator in float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension (input channels)
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K
        
        # Load A (weight): A[m_offsets, k_offsets] -> [BLOCK_M, BLOCK_K]
        a_ptrs = a_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_vals = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Load B (input): B[batch_id, k_offsets, n_offsets] -> [BLOCK_K, BLOCK_N]
        b_ptrs = b_ptr + batch_id * stride_batch_b + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        b_vals = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        
        # acc += A @ B
        acc += tl.dot(a_vals, b_vals)
    
    # Add bias
    acc += bias[:, None]
    
    # Store C (output): C[batch_id, m_offsets, n_offsets]
    c_ptrs = c_ptr + batch_id * stride_batch_c + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])


@torch.fx.wrap
def fused_conv2d_1x1_flatten(bias, weight, input_tensor):
    # 1x1 conv2d + flatten(dim=2) fused kernel
    # input_tensor: [N, C_in, H, W] - treat as [N, C_in, HW] using strides
    # weight: [C_out, C_in, 1, 1] - treat as [C_out, C_in] using strides  
    # bias: [C_out]
    # output: [N, C_out, HW]
    
    N = input_tensor.shape[0]
    C_in = input_tensor.shape[1]
    H = input_tensor.shape[2]
    W = input_tensor.shape[3]
    C_out = weight.shape[0]
    HW = H * W
    
    # Create output tensor
    output = torch.empty((N, C_out, HW), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Weight strides: [C_out, C_in, 1, 1]
    # The last two dims are size 1, so stride(2) and stride(3) don't matter
    # We treat it as [C_out, C_in] with stride_am = stride(0), stride_ak = stride(1)
    stride_am = weight.stride(0)
    stride_ak = weight.stride(1)
    
    # Input strides: [N, C_in, H, W]
    # We treat it as [N, C_in, HW] where HW = H*W
    # stride_batch_b = stride(0) = C_in * H * W
    # stride_bk = stride(1) = H * W
    # stride_bn = stride(3) = 1 (since W stride for contiguous tensor)
    # For contiguous input: stride(0)=C_in*H*W, stride(1)=H*W, stride(2)=W, stride(3)=1
    # When treating as [N, C_in, HW], accessing element [n, c_in, hw]:
    # offset = n * stride(0) + c_in * stride(1) + hw * 1
    # But hw = h * W + w, and in original tensor it's h * stride(2) + w * stride(3) = h * W + w * 1
    # For contiguous: h * W + w = hw, so stride for the HW dim is 1, which equals stride(3)
    stride_batch_b = input_tensor.stride(0)
    stride_bk = input_tensor.stride(1)
    stride_bn = input_tensor.stride(3)  # W stride = 1 for contiguous
    
    # Output strides: [N, C_out, HW]
    stride_cm = output.stride(1)  # HW
    stride_cn = output.stride(2)  # 1
    stride_batch_c = output.stride(0)  # C_out * HW
    
    M = C_out
    NK = HW
    K = C_in
    
    BLOCK_M = 16
    BLOCK_N = 64
    BLOCK_K = 32
    GROUP_M = 8
    
    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(NK, BLOCK_N)
    grid_size = N * num_pid_m * num_pid_n
    
    conv1x1_flatten_kernel[
        (grid_size,)
    ](
        a_ptr=weight,
        b_ptr=input_tensor,
        bias_ptr=bias,
        c_ptr=output,
        M=M, N=NK, K=K,
        stride_am=stride_am,
        stride_ak=stride_ak,
        stride_bk=stride_bk,
        stride_bn=stride_bn,
        stride_cm=stride_cm,
        stride_cn=stride_cn,
        stride_batch_b=stride_batch_b,
        stride_batch_c=stride_batch_c,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
    )
    
    return output


def replacement_func():
    return fused_conv2d_1x1_flatten