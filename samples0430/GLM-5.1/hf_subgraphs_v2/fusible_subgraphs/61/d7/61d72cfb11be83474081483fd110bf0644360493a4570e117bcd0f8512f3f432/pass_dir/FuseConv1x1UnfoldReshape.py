import torch
import triton
import triton.language as tl
import math

# Pattern matching function - mirrors model.py exactly
def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.nn.functional.unfold(conv2d, kernel_size=(2, 2), stride=(2, 2))
    tmp_3 = tmp_2.reshape(1, 128, 4, -1)
    return (tmp_3,)

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ============================================================
# Triton kernels for fused computation
# ============================================================

# Kernel 1: Matmul kernel for 1x1 conv (W @ X -> intermediate)
# A = weight [128, 256], B = input [256, 1024], C = intermediate [128, 1024]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 4}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Grouped program ordering for L2 cache optimization
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size
    pid_n = (pid % num_pid_in_group) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load A tile: weight [BLOCK_M, BLOCK_K]
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile: input [BLOCK_K, BLOCK_N]
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Matrix multiply using tensor cores
        acc += tl.dot(a, b, allow_tf32=True)

    # Store result with dtype conversion
    offs_m_final = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_final = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_m_final[:, None] * stride_cm + offs_n_final[None, :] * stride_cn
    c_mask = (offs_m_final[:, None] < M) & (offs_n_final[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# Kernel 2: Rearrange kernel (unfold + reshape)
# Reads from intermediate [1, C, H, W] and writes to output [1, C, 4, N_BLOCKS]
# output[c, k, n] = intermediate[c, 2*(n//16) + k//2, 2*(n%16) + k%2]
@triton.jit
def rearrange_kernel(
    inter_ptr, output_ptr,
    C, H, W, N_BLOCKS,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = C * 4 * N_BLOCKS
    mask = offsets < total

    # Decode flat output index into (c, k, n)
    c_idx = offsets // (4 * N_BLOCKS)
    rem = offsets % (4 * N_BLOCKS)
    k_idx = rem // N_BLOCKS
    n_idx = rem % N_BLOCKS

    # Compute source spatial position from (k, n)
    bh = n_idx // 16
    bw = n_idx % 16
    ky = k_idx // 2
    kx = k_idx % 2
    h = 2 * bh + ky
    w = 2 * bw + kx

    # Source offset in intermediate tensor [1, C, H, W]
    # intermediate[0, c, h, w] at offset c * H * W + h * W + w
    src_offset = c_idx * (H * W) + h * W + w

    vals = tl.load(inter_ptr + src_offset, mask=mask)
    tl.store(output_ptr + offsets, vals, mask=mask)


# ============================================================
# Kernel wrapper (decorated with @torch.fx.wrap)
# ============================================================
@torch.fx.wrap
def fused_conv_unfold_reshape(weight, input_tensor):
    """
    Fused implementation of: 1x1 conv + unfold(2x2, stride 2) + reshape(1, 128, 4, -1)
    
    Input:
      weight: [128, 256, 1, 1] - convolution weight
      input_tensor: [1, 256, 32, 32] - input feature map
    
    Output:
      [1, 128, 4, 256] - unfolded and reshaped result
    """
    C = weight.shape[0]   # 128
    IC = weight.shape[1]  # 256
    H = input_tensor.shape[2]  # 32
    W_dim = input_tensor.shape[3]  # 32
    N = H * W_dim  # 1024 (total spatial positions)
    N_BLOCKS = (H // 2) * (W_dim // 2)  # 256 (2x2 patches with stride 2)

    # Step 1: Allocate intermediate for matmul result [1, C, H, W_dim]
    intermediate = torch.empty((1, C, H, W_dim), dtype=input_tensor.dtype, device=input_tensor.device)

    # Step 2: Run matmul kernel (1x1 conv)
    # W[C, IC] @ X[IC, N] -> intermediate[C, N]
    M = C  # 128
    K = IC  # 256

    grid_m = math.ceil(M / 32)  # Use default block sizes for grid calculation
    grid_n = math.ceil(N / 32)
    grid = (grid_m * grid_n,)

    conv1x1_matmul_kernel[grid](
        a_ptr=weight, b_ptr=input_tensor, c_ptr=intermediate,
        M=M, N=N, K=K,
        stride_am=IC, stride_ak=1,
        stride_bk=H * W_dim, stride_bn=1,
        stride_cm=H * W_dim, stride_cn=1,
    )

    # Step 3: Allocate output [1, C, 4, N_BLOCKS]
    output = torch.empty((1, C, 4, N_BLOCKS), dtype=input_tensor.dtype, device=input_tensor.device)

    # Step 4: Run rearrange kernel
    total_elements = C * 4 * N_BLOCKS
    BLOCK_SIZE = 1024
    grid_rearrange = (math.ceil(total_elements / BLOCK_SIZE),)

    rearrange_kernel[grid_rearrange](
        inter_ptr=intermediate, output_ptr=output,
        C=C, H=H, W=W_dim, N_BLOCKS=N_BLOCKS,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


# Replacement function (returns the kernel wrapper function)
def replacement_func():
    return fused_conv_unfold_reshape