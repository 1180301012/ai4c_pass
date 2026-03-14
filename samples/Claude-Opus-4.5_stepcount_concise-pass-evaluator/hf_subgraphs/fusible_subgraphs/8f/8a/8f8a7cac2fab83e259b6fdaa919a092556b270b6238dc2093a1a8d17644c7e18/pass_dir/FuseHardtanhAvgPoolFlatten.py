import torch
import triton
import triton.language as tl


def pattern(in_0, batch_size):
    """
    Pattern to match hardtanh + adaptive_avg_pool2d + view + flatten
    """
    tmp_0 = torch.nn.functional.hardtanh(in_0, 0.0, 6.0, True)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    tmp_2 = tmp_1.view(batch_size, -1)
    tmp_3 = torch.flatten(tmp_2, 1)
    return tmp_3


def replacement_args(in_0, batch_size):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 32}, num_warps=1),
        triton.Config({'BLOCK_HW': 64}, num_warps=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 512}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def fused_hardtanh_avgpool_coalesced(
    input_ptr,
    output_ptr,
    NC,  # N * C
    HW,  # H * W
    inv_HW,
    BLOCK_HW: tl.constexpr,
):
    """
    Coalesced memory access kernel.
    Grid: (N * C,)
    Each program handles one (batch, channel) pair and reads HW contiguous elements.
    """
    # Program ID is the linear index into (N, C)
    pid = tl.program_id(0)
    
    # Base offset for this (batch, channel) - data is contiguous in HW
    base_offset = pid * HW
    
    # Accumulate sum over spatial dimensions (contiguous load)
    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)
    
    for start in range(0, HW, BLOCK_HW):
        offsets = start + tl.arange(0, BLOCK_HW)
        mask = offsets < HW
        
        # Contiguous memory access!
        x = tl.load(input_ptr + base_offset + offsets, mask=mask, other=0.0)
        
        # Apply hardtanh: clamp to [0, 6]
        x = tl.maximum(x, 0.0)
        x = tl.minimum(x, 6.0)
        
        acc += tl.where(mask, x, 0.0)
    
    # Sum up and compute mean
    total_sum = tl.sum(acc, axis=0)
    mean_val = total_sum * inv_HW
    
    # Store result
    tl.store(output_ptr + pid, mean_val)


@torch.fx.wrap
def fused_hardtanh_avgpool_flatten(in_0):
    """
    Wrapper function that launches the fused kernel.
    """
    N, C, H, W = in_0.shape
    HW = H * W
    inv_HW = 1.0 / HW
    NC = N * C
    
    # Output tensor with shape [N, C]
    output = torch.empty((N, C), dtype=in_0.dtype, device=in_0.device)
    
    # Launch one program per (batch, channel) pair
    grid = (NC,)
    
    fused_hardtanh_avgpool_coalesced[grid](
        in_0,
        output,
        NC,
        HW,
        inv_HW,
    )
    
    return output


def replacement_func():
    return fused_hardtanh_avgpool_flatten