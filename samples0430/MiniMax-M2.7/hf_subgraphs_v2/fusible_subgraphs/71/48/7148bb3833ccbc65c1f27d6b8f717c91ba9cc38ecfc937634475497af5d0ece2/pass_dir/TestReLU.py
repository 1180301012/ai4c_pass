import torch
import triton
import triton.language as tl


@triton.jit
def relu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Triton kernel for ReLU"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.where(x > 0, x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def triton_relu(x):
    """Triton implementation of ReLU"""
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    relu_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


@triton.jit
def fused_residual_gate_kernel(
    in_2_ptr,
    in_0_ptr,
    out_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    in_2_stride_n: tl.constexpr,
    in_2_stride_c: tl.constexpr,
    in_2_stride_h: tl.constexpr,
    in_2_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for relu + avg_pool + subtract + multiply + add"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    total_elements = N * C * H * W
    mask = offsets < total_elements
    
    n = offsets // (C * H * W)
    remainder = offsets % (C * H * W)
    c = remainder // (H * W)
    remainder2 = remainder % (H * W)
    h = remainder2 // W
    w = remainder2 % W
    
    # Load input
    in_2_offset = n * in_2_stride_n + c * in_2_stride_c + h * in_2_stride_h + w * in_2_stride_w
    x = tl.load(in_2_ptr + in_2_offset, mask=mask, other=0.0)
    
    # ReLU
    x_relu = tl.where(x > 0, x, 0.0)
    
    # Avg pooling 3x3 with padding=1
    pool_sum = tl.constexpr(0.0)
    pool_count = 0
    
    for kh in range(-1, 2):
        for kw in range(-1, 2):
            h_pool = h + kh
            w_pool = w + kw
            h_valid = (h_pool >= 0) & (h_pool < H)
            w_valid = (w_pool >= 0) & (w_pool < W)
            valid = h_valid & w_valid
            
            if valid:
                pool_offset = n * in_2_stride_n + c * in_2_stride_c + h_pool * in_2_stride_h + w_pool * in_2_stride_w
                pool_val = tl.load(in_2_ptr + pool_offset, mask=mask, other=0.0)
                pool_val_relu = tl.where(pool_val > 0, pool_val, 0.0)
                pool_sum = pool_sum + pool_val_relu
                pool_count = pool_count + 1
    
    pooled_val = pool_sum / pool_count.to(tl.float32)
    
    # Subtract
    diff = pooled_val - x_relu
    
    # Load weight and multiply
    weight = tl.load(in_0_ptr + c)
    product = weight * diff
    
    # Add
    result = x_relu + product
    
    # Store
    out_offset = n * out_stride_n + c * out_stride_c + h * out_stride_h + w * out_stride_w
    tl.store(out_ptr + out_offset, result, mask=mask)


@torch.fx.wrap
def fused_residual_gate(in_0, in_1, in_2):
    """Fused implementation"""
    N, C, H, W = in_2.shape
    out = torch.empty_like(in_2)
    
    BLOCK_SIZE = 1024
    num_programs = N * C * H * W
    
    in_2_stride_n = in_2.stride(0)
    in_2_stride_c = in_2.stride(1)
    in_2_stride_h = in_2.stride(2)
    in_2_stride_w = in_2.stride(3)
    out_stride_n = out.stride(0)
    out_stride_c = out.stride(1)
    out_stride_h = out.stride(2)
    out_stride_w = out.stride(3)
    
    grid = (num_programs,)
    
    fused_residual_gate_kernel[grid](
        in_2, in_0, out,
        N, C, H, W,
        in_2_stride_n, in_2_stride_c, in_2_stride_h, in_2_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        BLOCK_SIZE,
    )
    
    return out


@triton.jit
def unsqueeze_expand_kernel(in_ptr, out_ptr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """Expand [C] to [C, H, W]"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < C * H * W
    
    c = offsets // (H * W)
    out_stride_c = H * W
    out_stride_h = W
    
    val = tl.load(in_ptr + c)
    out_offset = c * out_stride_c + ((offsets % (H * W)) // W) * out_stride_h + (offsets % W)
    
    tl.store(out_ptr + out_offset, val, mask=mask)


@torch.fx.wrap
def unsqueeze_expand_1d_to_3d(in_1, H, W):
    """Expand 1D tensor to 3D"""
    C = in_1.shape[0]
    out = torch.empty((C, H, W), dtype=in_1.dtype, device=in_1.device)
    
    BLOCK_SIZE = 1024
    num_programs = C * H * W
    grid = (num_programs,)
    
    unsqueeze_expand_kernel[grid](in_1, out, C, H, W, BLOCK_SIZE)
    
    return out


@torch.fx.wrap
def fused_residual_gate_wrapper(in_0, in_1, in_2):
    """Wrapper for fused computation"""
    tmp_8 = fused_residual_gate(in_0, in_1, in_2)
    N, C, H, W = in_2.shape
    tmp_10 = unsqueeze_expand_1d_to_3d(in_1, 1, 1)
    return (tmp_8, tmp_10)


def pattern(x, y):
    """Simple add pattern"""
    return x + y


def replacement_args(x, y):
    return (x, y)


def replacement_func():
    return triton_add


@triton.jit
def triton_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def triton_add(x, y):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(x)
    triton_add_kernel[(num_programs,)](x, y, out, N, BLOCK_SIZE)
    return out