import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.nn.functional.pad(conv2d, [2, 2, 2, 2], 'constant', None)
    tmp_3 = tmp_2.unfold(2, 12, 8)
    tmp_4 = tmp_3.unfold(3, 12, 8)
    tmp_5 = tmp_4.reshape(8, 80, 4, -1)
    tmp_6 = tmp_5.permute(0, 2, 3, 1)
    split = torch.functional.split(tmp_6, [16, 64], dim = -1)
    tmp_8 = split[0]
    tmp_9 = split[1]
    tmp_10 = tmp_8.transpose(-1, -2)
    return (tmp_10, tmp_9)


def replacement_args(in_0, in_1):
    return (in_0, in_1, "route_float16")


# ============ Triton Matmul Kernel (1x1 Conv) ============

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
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
):
    pid = tl.program_id(0)
    num_pid_m = (M + BLOCK_M - 1) // BLOCK_M
    num_pid_n = (N + BLOCK_N - 1) // BLOCK_N
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < N

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                     mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                     mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        accumulator += tl.dot(a, b, allow_tf32=True)

    c = accumulator.to(c_ptr.dtype.element_ty)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             c, mask=m_mask[:, None] & n_mask[None, :])


# ============ Triton Reorganization Kernel ============

@triton.jit
def reorg_kernel(
    conv_out_ptr,
    out_ptr,
    C_out, H, W,
    num_heads, head_dim_k, head_dim_v,
    patch_size, num_patches, pad_size,
    total_elements,
    OUTPUT_TYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    patch_area = patch_size * patch_size

    if OUTPUT_TYPE == 0:
        # out_k shape: [num_heads, num_patches, head_dim_k, patch_area]
        pos = offsets % patch_area
        dim_k = (offsets // patch_area) % head_dim_k
        patch = (offsets // (patch_area * head_dim_k)) % num_patches
        head = offsets // (patch_area * head_dim_k * num_patches)

        px = pos % patch_size
        py = pos // patch_size

        channel = head * (head_dim_k + head_dim_v) + dim_k
    else:
        # out_v shape: [num_heads, num_patches, patch_area, head_dim_v]
        dim_v = offsets % head_dim_v
        pos = (offsets // head_dim_v) % patch_area
        patch = (offsets // (head_dim_v * patch_area)) % num_patches
        head = offsets // (head_dim_v * patch_area * num_patches)

        px = pos % patch_size
        py = pos // patch_size

        channel = head * (head_dim_k + head_dim_v) + head_dim_k + dim_v

    # Map patch to padded starting position
    ph_start = tl.where(patch < 2, 0, 8)
    pw_start = tl.where((patch % 2) == 0, 0, 8)

    h_padded = ph_start + py
    w_padded = pw_start + px
    h_orig = h_padded - pad_size
    w_orig = w_padded - pad_size

    # Check if position is in valid conv output region (not in padding)
    valid = (h_orig >= 0) & (h_orig < H) & (w_orig >= 0) & (w_orig < W) & mask & (channel < C_out)

    # Compute offset in conv output [C_out, H*W]
    conv_offset = channel * (H * W) + h_orig * W + w_orig

    # Load from conv output (0 if in padding region)
    value = tl.load(conv_out_ptr + conv_offset, mask=valid, other=0.0)

    # Store to output
    tl.store(out_ptr + offsets, value, mask=mask)


# ============ Dispatch Wrapper ============

@torch.fx.wrap
def fused_conv_unfold_split_transpose(weight, input_tensor, route):
    if route == "route_float16":
        num_heads = 8
        head_dim_k = 16
        head_dim_v = 64
    elif route == "route_bfloat16":
        num_heads = 8
        head_dim_k = 16
        head_dim_v = 32
    elif route == "route_float32":
        num_heads = 8
        head_dim_k = 16
        head_dim_v = 32
    else:
        raise ValueError(f"Unknown route: {route}")

    patch_size = 12
    num_patches = 4
    pad_size = 2

    C_out = weight.shape[0]
    C_in = weight.shape[1]
    H = input_tensor.shape[2]
    W = input_tensor.shape[3]
    dtype = input_tensor.dtype
    device = input_tensor.device

    # Phase 1: Compute 1x1 conv as matmul
    M = C_out
    N = H * W
    K = C_in

    conv_out = torch.empty([M, N], dtype=dtype, device=device)

    grid = lambda META: (
        (M + META['BLOCK_M'] - 1) // META['BLOCK_M'] *
        (N + META['BLOCK_N'] - 1) // META['BLOCK_N'],
    )

    conv1x1_matmul_kernel[grid](
        weight.data_ptr(), input_tensor.data_ptr(), conv_out.data_ptr(),
        M, N, K,
        weight.stride(0), weight.stride(1),
        input_tensor.stride(1), input_tensor.stride(3),
        N, 1,
    )

    # Phase 2: Reorganize conv output into final format
    patch_area = patch_size * patch_size
    out_k = torch.empty([num_heads, num_patches, head_dim_k, patch_area], dtype=dtype, device=device)
    out_v = torch.empty([num_heads, num_patches, patch_area, head_dim_v], dtype=dtype, device=device)

    total_k = num_heads * num_patches * head_dim_k * patch_area
    total_v = num_heads * num_patches * patch_area * head_dim_v

    BLOCK_SIZE = 512

    grid_k = ((total_k + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    reorg_kernel[grid_k](
        conv_out.data_ptr(),
        out_k.data_ptr(),
        C_out, H, W,
        num_heads, head_dim_k, head_dim_v,
        patch_size, num_patches, pad_size,
        total_k,
        OUTPUT_TYPE=0,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    grid_v = ((total_v + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    reorg_kernel[grid_v](
        conv_out.data_ptr(),
        out_v.data_ptr(),
        C_out, H, W,
        num_heads, head_dim_k, head_dim_v,
        patch_size, num_patches, pad_size,
        total_v,
        OUTPUT_TYPE=1,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out_k, out_v


def replacement_func():
    return fused_conv_unfold_split_transpose