import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    """
    Match the pattern: Conv2D (1x1) -> View -> Softmax
    Must match the exact structure in model.py with batch size 32
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = tmp_2.view(32, 1, -1)
    tmp_4 = tmp_3.softmax(dim=-1)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['spatial_size', 'in_channels'],
)
@triton.jit
def fused_conv_kernel_b32(
    input_ptr,
    weight_ptr,
    bias_ptr,
    temp_ptr,
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    spatial_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute 1x1 conv in parallel across batches and spatial locations
    """
    batch_idx = tl.program_id(0)
    spatial_block_idx = tl.program_id(1)
    
    spatial_offsets = spatial_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    spatial_mask = spatial_offsets < spatial_size
    
    # Compute 1x1 convolution
    conv_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for c in range(in_channels):
        input_offset = batch_idx * in_channels * spatial_size + c * spatial_size + spatial_offsets
        inp = tl.load(input_ptr + input_offset, mask=spatial_mask, other=0.0)
        w = tl.load(weight_ptr + c)
        conv_val += inp * w
    
    bias = tl.load(bias_ptr)
    conv_val += bias
    
    # Store to temp
    temp_offset = batch_idx * spatial_size + spatial_offsets
    tl.store(temp_ptr + temp_offset, conv_val, mask=spatial_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['spatial_size'],
)
@triton.jit
def softmax_kernel_b32(
    temp_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    spatial_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Apply softmax per batch
    """
    batch_idx = tl.program_id(0)
    
    temp_base = temp_ptr + batch_idx * spatial_size
    output_base = output_ptr + batch_idx * spatial_size
    
    # Find max
    max_val = float('-inf')
    num_blocks = tl.cdiv(spatial_size, BLOCK_SIZE)
    for i in range(num_blocks):
        offs = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        m = offs < spatial_size
        vals = tl.load(temp_base + offs, mask=m, other=float('-inf'))
        max_val = tl.maximum(max_val, tl.max(vals))
    
    # Sum of exp
    sum_exp = 0.0
    for i in range(num_blocks):
        offs = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        m = offs < spatial_size
        vals = tl.load(temp_base + offs, mask=m, other=0.0)
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(tl.where(m, exp_vals, 0.0))
    
    # Normalize
    for i in range(num_blocks):
        offs = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        m = offs < spatial_size
        vals = tl.load(temp_base + offs, mask=m, other=0.0)
        softmax_vals = tl.exp(vals - max_val) / sum_exp
        tl.store(output_base + offs, softmax_vals, mask=m)


@torch.fx.wrap
def fused_conv2d_view_softmax_batch32(in_0, in_1, in_2):
    """
    Fused Conv2D + View + Softmax - with autotuning
    """
    bias = in_0
    weight = in_1
    input_tensor = in_2
    
    batch_size, in_channels, height, width = input_tensor.shape
    spatial_size = height * width
    
    # Allocate buffers
    temp = torch.empty((batch_size, 1, spatial_size), dtype=input_tensor.dtype, device=input_tensor.device)
    output = torch.empty((batch_size, 1, spatial_size), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Step 1: Compute conv in parallel across all batches
    BLOCK_SIZE = 1024
    num_spatial_blocks = (spatial_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    fused_conv_kernel_b32[(batch_size, num_spatial_blocks)](
        input_tensor, weight, bias, temp,
        batch_size=batch_size,
        in_channels=in_channels,
        spatial_size=spatial_size,
    )
    
    # Step 2: Apply softmax per batch
    softmax_kernel_b32[(batch_size,)](
        temp, output,
        batch_size=batch_size,
        spatial_size=spatial_size,
    )
    
    return output


def replacement_func():
    return fused_conv2d_view_softmax_batch32