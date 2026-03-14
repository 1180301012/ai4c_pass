import torch
import triton
import triton.language as tl


def pattern(in_0):
    """
    Fuse permute + view for Graph 1 (196 -> 14x14)
    """
    tmp_0 = in_0.permute(0, 2, 1)
    tmp_1 = tmp_0.view(1, 384, 14, 14)
    return tmp_1


def replacement_args(in_0):
    return (in_0,)


# Autotune configurations for better performance
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=4, num_warps=4),
    ],
    key=['num_elements'],
)
@triton.jit
def fused_permute_view_kernel_14(
    in_ptr, out_ptr,
    C: tl.constexpr,
    num_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Fused kernel for permute + view (14x14 case)
    Input: [1, 196, 384]
    Output: [1, 384, 14, 14]
    """
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    
    # Convert linear offset to (c, i, j) for view
    # out shape: [C=384, 14, 14], flat index = c * 196 + i * 14 + j
    c = offsets // 196
    rem = offsets % 196
    i = rem // 14
    j = rem % 14
    
    # Map to input indices: after permute, input is [1, 384, 196]
    # Input index mapping: output[c, i, j] corresponds to input[0, c, i*14 + j]
    # After permute (0,2,1), input[0, i*14+j, c] goes to output[0, c, i*14+j]
    # So we need input[0, i*14+j, c]
    in_dim1 = i * 14 + j
    in_offset = in_dim1 * C + c
    
    vals = tl.load(in_ptr + in_offset, mask=mask, other=0.0)
    
    # Store to out
    out_offset = c * 196 + i * 14 + j
    tl.store(out_ptr + out_offset, vals, mask=mask)


@torch.fx.wrap
def fused_permute_view_14(in_0):
    """
    Fused permute + view for 14x14 case
    Input: [1, 196, 384]
    Output: [1, 384, 14, 14]
    """
    B, N, C = in_0.shape  # [1, 196, 384]
    
    # Output tensor
    out = torch.empty((1, C, 14, 14), dtype=in_0.dtype, device=in_0.device)
    
    # Flatten for easier addressing
    in_flat = in_0.view(-1)
    out_flat = out.view(-1)
    
    num_elements = C * 14 * 14  # 75264
    
    # With autotune, we launch with grid = (num_elements,) and Triton
    # will autotune the BLOCK_SIZE
    grid = (num_elements,)
    
    fused_permute_view_kernel_14[grid](
        in_ptr=in_flat,
        out_ptr=out_flat,
        C=C,
        num_elements=num_elements,
        BLOCK_SIZE=256  # Initial value, will be autotuned
    )
    
    return out


def replacement_func():
    return fused_permute_view_14