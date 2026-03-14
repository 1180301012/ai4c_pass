import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """Pattern to match: multiply, sum along dim 1, unsqueeze, sigmoid"""
    tmp_0 = in_1 * in_0
    tmp_1 = torch.sum(tmp_0, dim=1)
    tmp_2 = tmp_1.unsqueeze(1)
    tmp_3 = torch.sigmoid(tmp_2)
    return tmp_3


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=1),
    ],
    key=['n_elements', 'channels'],
)
@triton.jit
def fused_mul_sum_sigmoid_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Process spatial elements in parallel
    pid = tl.program_id(0)
    
    # Each block handles BLOCK_SIZE spatial elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Decode offsets to (batch, h, w)
    hw_size = height * width
    batch_ids = offsets // hw_size
    hw_offsets = offsets % hw_size
    
    # Initialize accumulator for each spatial element
    accs = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Loop over channels
    for c in range(channels):
        # Compute memory offsets: [B, C, H, W] layout
        # offset = b * (C*H*W) + c * (H*W) + hw
        mem_offsets = batch_ids * channels * hw_size + c * hw_size + hw_offsets
        
        # Load and multiply
        val_0 = tl.load(in_0_ptr + mem_offsets, mask=mask, other=0.0)
        val_1 = tl.load(in_1_ptr + mem_offsets, mask=mask, other=0.0)
        
        # Accumulate
        accs += val_0 * val_1
    
    # Apply sigmoid
    results = 1.0 / (1.0 + tl.exp(-accs))
    
    # Store results
    out_offsets = batch_ids * hw_size + hw_offsets
    tl.store(out_ptr + out_offsets, results, mask=mask)


@torch.fx.wrap
def fused_mul_sum_sigmoid(in_0, in_1):
    batch_size, channels, height, width = in_0.shape
    n_elements = batch_size * height * width
    
    # Output shape: [batch_size, 1, height, width]
    out = torch.empty(batch_size, 1, height, width, device=in_0.device, dtype=in_0.dtype)
    
    # Grid: 1D over all spatial elements
    grid = lambda META: (triton.cdiv(n_elements, META['BLOCK_SIZE']),)
    
    fused_mul_sum_sigmoid_kernel[grid](
        in_0,
        in_1,
        out,
        batch_size,
        channels,
        height,
        width,
        n_elements,
    )
    
    return out


def replacement_func():
    return fused_mul_sum_sigmoid