import torch
import triton
import triton.language as tl

# Constants for fast GELU approximation
GELU_CONST = 0.7978845608028654  # sqrt(2/pi)
GELU_SCALE = 0.5
GELU_PRECONST = 0.044715


@triton.jit
def gelu_kernel(
    x_ptr,
    output_ptr,
    mean_ptr,
    b_strides,
    c_strides,
    h_strides,
    w_strides,
    n_b,
    n_c,
    n_h,
    n_w,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    num_warps: tl.constexpr,
):
    """
    Fused GELU + mean over spatial dimensions (H, W).
    Returns both the full GELU output and the reduced mean.
    
    Uses a 2D blocking strategy over (H, W) with reduction across threads.
    """
    # Get program ids
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    # Get offsets for H dimension - each program handles a block of H
    h_offset_base = tl.program_id(2) * BLOCK_SIZE_H
    
    # Local accumulator for mean
    local_sum = tl.zeros((BLOCK_SIZE_H,), dtype=tl.float32)
    
    # Iterate over W dimension
    for w in range(n_w):
        # Load data - use mask for boundary checks
        h_offsets = h_offset_base + tl.arange(0, BLOCK_SIZE_H)
        w_offset = w
        
        # Calculate linear offsets
        offsets = (
            pid_b * b_strides[0] +
            pid_c * c_strides[0] +
            h_offsets * h_strides[0] +
            w_offset * w_strides[0]
        )
        
        # Create masks
        mask_h = h_offsets < n_h
        
        # Load x
        x = tl.load(x_ptr + offsets, mask=mask_h, other=0.0)
        
        # Compute GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        x_cubed = x * x * x
        gelu_arg = GELU_CONST * (x + GELU_PRECONST * x_cubed)
        gelu = GELU_SCALE * x * (1.0 + tl.math.tanh(gelu_arg))
        
        # Store GELU output
        tl.store(output_ptr + offsets, gelu, mask=mask_h)
        
        # Accumulate for mean (sum over spatial dims)
        local_sum = local_sum + tl.sum(gelu, axis=0)
    
    # Reduction across W dimension
    local_sum = tl.sum(local_sum, axis=0)
    
    # Calculate number of elements for mean
    num_elements = n_h * n_w
    mean_val = local_sum / tl.cast(num_elements, tl.float32)
    
    # Store mean output - shape is [B, C, 1, 1]
    mean_offset = pid_b * n_c + pid_c
    tl.store(mean_ptr + mean_offset, mean_val)


@triton.jit
def gelu_kernel_autotuned(
    x_ptr,
    output_ptr,
    mean_ptr,
    b_strides,
    c_strides,
    h_strides,
    w_strides,
    n_b,
    n_c,
    n_h,
    n_w,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    """
    Autotuned GELU + mean kernel with autotuning support.
    """
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    h_offset_base = tl.program_id(2) * BLOCK_SIZE_H
    
    local_sum = tl.zeros((BLOCK_SIZE_H,), dtype=tl.float32)
    
    for w in range(n_w):
        h_offsets = h_offset_base + tl.arange(0, BLOCK_SIZE_H)
        
        offsets = (
            pid_b * b_strides[0] +
            pid_c * c_strides[0] +
            h_offsets * h_strides[0] +
            w * w_strides[0]
        )
        
        mask_h = h_offsets < n_h
        x = tl.load(x_ptr + offsets, mask=mask_h, other=0.0)
        
        # Fast GELU
        x_cubed = x * x * x
        gelu_arg = GELU_CONST * (x + GELU_PRECONST * x_cubed)
        gelu = GELU_SCALE * x * (1.0 + tl.math.tanh(gelu_arg))
        
        tl.store(output_ptr + offsets, gelu, mask=mask_h)
        local_sum = local_sum + tl.sum(gelu, axis=0)
    
    local_sum = tl.sum(local_sum, axis=0)
    num_elements = n_h * n_w
    mean_val = local_sum / tl.cast(num_elements, tl.float32)
    
    mean_offset = pid_b * n_c + pid_c
    tl.store(mean_ptr + mean_offset, mean_val)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_H': 8, 'BLOCK_SIZE_W': 1}, num_warps=2),
        triton.Config({'BLOCK_SIZE_H': 16, 'BLOCK_SIZE_W': 1}, num_warps=4),
        triton.Config({'BLOCK_SIZE_H': 32, 'BLOCK_SIZE_W': 1}, num_warps=4),
        triton.Config({'BLOCK_SIZE_H': 8, 'BLOCK_SIZE_W': 1}, num_warps=4, pre_hook=None),
        triton.Config({'BLOCK_SIZE_H': 16, 'BLOCK_SIZE_W': 1}, num_warps=8),
        triton.Config({'BLOCK_SIZE_H': 32, 'BLOCK_SIZE_W': 1}, num_warps=8),
    ],
    key=['n_h', 'n_w'],
)
@triton.jit
def gelu_mean_fused_kernel(
    x_ptr,
    output_ptr,
    mean_ptr,
    b_strides,
    c_strides,
    h_strides,
    w_strides,
    n_b,
    n_c,
    n_h,
    n_w,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    """
    Optimized fused GELU + mean over spatial dimensions (H, W).
    """
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    h_offset_base = tl.program_id(2) * BLOCK_SIZE_H
    
    local_sum = tl.zeros((BLOCK_SIZE_H,), dtype=tl.float32)
    
    for w in range(n_w):
        h_offsets = h_offset_base + tl.arange(0, BLOCK_SIZE_H)
        
        offsets = (
            pid_b * b_strides[0] +
            pid_c * c_strides[0] +
            h_offsets * h_strides[0] +
            w * w_strides[0]
        )
        
        mask_h = h_offsets < n_h
        x = tl.load(x_ptr + offsets, mask=mask_h, other=0.0)
        
        # Fast GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        x_cubed = x * x * x
        gelu_arg = GELU_CONST * (x + GELU_PRECONST * x_cubed)
        gelu = GELU_SCALE * x * (1.0 + tl.math.tanh(gelu_arg))
        
        tl.store(output_ptr + offsets, gelu, mask=mask_h)
        local_sum = local_sum + tl.sum(gelu, axis=0)
    
    local_sum = tl.sum(local_sum, axis=0)
    num_elements = n_h * n_w
    mean_val = local_sum / tl.cast(num_elements, tl.float32)
    
    mean_offset = pid_b * n_c + pid_c
    tl.store(mean_ptr + mean_offset, mean_val)


@torch.fx.wrap
def gelu_mean_fused(x):
    """
    Fused GELU + mean over spatial dimensions.
    x: [B, C, H, W] tensor
    Returns: (gelu_output, mean_output)
    - gelu_output: [B, C, H, W]
    - mean_output: [B, C, 1, 1]
    """
    b, c, h, w = x.shape
    
    # Allocate output tensors
    output = torch.empty_like(x)
    mean_output = torch.empty((b, c, 1, 1), dtype=torch.float32, device=x.device)
    
    # Get strides
    b_stride = x.stride(0)
    c_stride = x.stride(1)
    h_stride = x.stride(2)
    w_stride = x.stride(3)
    
    # Grid configuration
    # We parallelize over B, C, and blocks of H
    BLOCK_SIZE_H = 32
    grid_h = (h + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    
    grid = (b, c, grid_h)
    
    # Convert strides to tensors for Triton
    strides_tensor = torch.tensor([b_stride, c_stride, h_stride, w_stride], dtype=torch.int64, device=x.device)
    
    # Launch kernel
    gelu_mean_fused_kernel[grid](
        x,
        output,
        mean_output,
        strides_tensor,
        strides_tensor,
        strides_tensor,
        strides_tensor,
        b,
        c,
        h,
        w,
        BLOCK_SIZE_H,
        1,
    )
    
    return output, mean_output


def pattern(x):
    """Match GELU followed by mean over spatial dimensions (2,3) with keepdim=True"""
    gelu_out = torch.nn.functional.gelu(x)
    mean_out = gelu_out.mean((2, 3), keepdim=True)
    return gelu_out, mean_out


def replacement_args(x):
    """Extract the input tensor"""
    return (x,)


def replacement_func():
    """Return the fused kernel function"""
    return gelu_mean_fused