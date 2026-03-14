import torch
import triton
import triton.language as tl

# Pattern to match: conv2d -> stack -> sum -> cat
# The stack([x], dim=0).sum(dim=0) is identity for single tensor - can be eliminated

def pattern(in_0, in_1, in_2, in_3):
    """
    Match the exact computation pattern from model.py
    in_0: bias [2048]
    in_1: weight [2048, 256, 1, 1]
    in_2: tensor to concatenate [batch, 2048, 64, 128]
    in_3: input to conv [batch, 256, 64, 128]
    """
    # Conv2d with 1x1 kernel
    tmp_2 = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    # Stack with single tensor (adds dim 0)
    tmp_3 = torch.stack([tmp_2], dim=0)
    # Sum along dim 0 (removes the added dim) - identity for single element!
    tmp_4 = tmp_3.sum(dim=0)
    # Concatenate along channel dimension
    tmp_5 = torch.cat([tmp_4, in_2], 1)
    return tmp_5


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.autotune(
    configs=[
        # Configurations optimized for C_in=256, C_out=2048, HW=8192
        triton.Config({'BLOCK_HW': 128, 'BLOCK_CO': 64, 'BLOCK_CI': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_HW': 64, 'BLOCK_CO': 128, 'BLOCK_CI': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_HW': 128, 'BLOCK_CO': 128, 'BLOCK_CI': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_HW': 256, 'BLOCK_CO': 64, 'BLOCK_CI': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_HW': 64, 'BLOCK_CO': 64, 'BLOCK_CI': 128}, num_stages=3, num_warps=4),
        # More aggressive configurations
        triton.Config({'BLOCK_HW': 256, 'BLOCK_CO': 128, 'BLOCK_CI': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_HW': 128, 'BLOCK_CO': 256, 'BLOCK_CI': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_HW': 64, 'BLOCK_CO': 256, 'BLOCK_CI': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_HW': 128, 'BLOCK_CO': 64, 'BLOCK_CI': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_HW': 64, 'BLOCK_CO': 64, 'BLOCK_CI': 256}, num_stages=2, num_warps=4),
        # Balanced configurations  
        triton.Config({'BLOCK_HW': 128, 'BLOCK_CO': 128, 'BLOCK_CI': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_HW': 64, 'BLOCK_CO': 128, 'BLOCK_CI': 128}, num_stages=2, num_warps=8),
        # Cover entire C_in in one iteration (C_in=256)
        triton.Config({'BLOCK_HW': 128, 'BLOCK_CO': 64, 'BLOCK_CI': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_HW': 64, 'BLOCK_CO': 128, 'BLOCK_CI': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_HW': 256, 'BLOCK_CO': 64, 'BLOCK_CI': 256}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_HW': 128, 'BLOCK_CO': 128, 'BLOCK_CI': 256}, num_stages=2, num_warps=8),
        # Different num_stages
        triton.Config({'BLOCK_HW': 64, 'BLOCK_CO': 128, 'BLOCK_CI': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_HW': 128, 'BLOCK_CO': 64, 'BLOCK_CI': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_HW': 64, 'BLOCK_CO': 64, 'BLOCK_CI': 128}, num_stages=4, num_warps=4),
    ],
    key=['N_batch', 'HW', 'C_out', 'C_in'],
)
@triton.jit
def fused_conv1x1_cat_kernel(
    input_ptr, weight_ptr, bias_ptr, in_2_ptr, output_ptr,
    N_batch, HW, C_out, C_in, C_b, C_total,
    BLOCK_HW: tl.constexpr, BLOCK_CO: tl.constexpr, BLOCK_CI: tl.constexpr,
):
    """
    Fused 1x1 convolution + concatenation kernel.
    Input: [N, C_in, HW]
    Weight: [C_out, C_in]
    in_2: [N, C_b, HW] - tensor to concatenate
    Output: [N, C_total, HW] where C_total = C_out + C_b
    
    The conv output goes to the first C_out channels of output.
    """
    # Program ID for batch, spatial, and output channel dimensions
    pid_n = tl.program_id(0)  # batch index
    pid_hw = tl.program_id(1)  # spatial block
    pid_co = tl.program_id(2)  # output channel block
    
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_HW, BLOCK_CO), dtype=tl.float32)
    
    # Precompute base pointers
    input_base = input_ptr + pid_n * (C_in * HW)
    
    # Main loop over input channels
    for ci in range(0, C_in, BLOCK_CI):
        offs_ci = ci + tl.arange(0, BLOCK_CI)
        
        # Load input tile [BLOCK_CI, BLOCK_HW]
        input_ptrs = input_base + offs_ci[:, None] * HW + offs_hw[None, :]
        input_mask = (offs_ci[:, None] < C_in) & (offs_hw[None, :] < HW)
        a = tl.load(input_ptrs, mask=input_mask, other=0.0)
        
        # Load weight tile [BLOCK_CO, BLOCK_CI]
        weight_ptrs = weight_ptr + offs_co[:, None] * C_in + offs_ci[None, :]
        weight_mask = (offs_co[:, None] < C_out) & (offs_ci[None, :] < C_in)
        b = tl.load(weight_ptrs, mask=weight_mask, other=0.0)
        
        # Accumulate: a.T @ b.T = [BLOCK_HW, BLOCK_CO]
        acc += tl.dot(tl.trans(a), tl.trans(b))
    
    # Load and add bias
    bias_mask = offs_co < C_out
    bias_val = tl.load(bias_ptr + offs_co, mask=bias_mask, other=0.0)
    acc = acc + bias_val[None, :]
    
    # Store output directly to first C_out channels of concatenated output
    # Output layout: [N, C_total, HW], conv goes to channels [0, C_out)
    output_ptrs = output_ptr + pid_n * (C_total * HW) + offs_co[None, :] * HW + offs_hw[:, None]
    output_mask = (offs_hw[:, None] < HW) & (offs_co[None, :] < C_out)
    tl.store(output_ptrs, acc, mask=output_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
        triton.Config({'BLOCK_SIZE': 8192}),
        triton.Config({'BLOCK_SIZE': 16384}),
    ],
    key=['total_elements'],
)
@triton.jit
def copy_kernel(
    src_ptr, dst_ptr,
    N, C_src, C_offset, C_total, HW,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Copy kernel - copies src tensor to dst tensor at channel offset.
    src: [N, C_src, HW]
    dst: [N, C_total, HW] - write to channels [C_offset, C_offset + C_src)
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    mask = offs < total_elements
    
    # Decompose source linear index to (n, c, hw)
    n = offs // (C_src * HW)
    remainder = offs % (C_src * HW)
    c = remainder // HW
    hw = remainder % HW
    
    # Load from source
    src_val = tl.load(src_ptr + offs, mask=mask, other=0.0)
    
    # Calculate destination index with channel offset
    dst_idx = n * (C_total * HW) + (c + C_offset) * HW + hw
    
    # Store to destination
    tl.store(dst_ptr + dst_idx, src_val, mask=mask)


@torch.fx.wrap
def optimized_conv_stack_sum_cat(bias, weight, in_2, in_3):
    """
    Optimized implementation using Triton:
    1. Skip redundant stack+sum (identity operation for single tensor)
    2. Implement 1x1 conv directly writing to concatenated output
    3. Copy in_2 to the second part of the output
    """
    # Get shapes
    N_batch, C_in, H, W = in_3.shape
    C_out = weight.shape[0]
    C_b = in_2.shape[1]
    HW = H * W
    C_total = C_out + C_b
    
    # Ensure contiguous
    in_3_contig = in_3.contiguous()
    weight_contig = weight.contiguous()
    in_2_contig = in_2.contiguous()
    
    # View weight as 2D: [C_out, C_in, 1, 1] -> [C_out, C_in]
    weight_2d = weight_contig.view(C_out, C_in)
    
    # Allocate final output [N, C_total, H, W]
    out = torch.empty((N_batch, C_total, H, W), device=in_3.device, dtype=in_3.dtype)
    
    # Launch fused conv kernel - writes to first C_out channels
    grid_conv = lambda meta: (
        N_batch,
        triton.cdiv(HW, meta['BLOCK_HW']),
        triton.cdiv(C_out, meta['BLOCK_CO']),
    )
    
    fused_conv1x1_cat_kernel[grid_conv](
        in_3_contig, weight_2d, bias, in_2_contig, out,
        N_batch, HW, C_out, C_in, C_b, C_total,
    )
    
    # Copy in_2 to channels [C_out, C_total)
    total_copy_elements = N_batch * C_b * HW
    grid_copy = lambda meta: ((total_copy_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    
    copy_kernel[grid_copy](
        in_2_contig, out,
        N_batch, C_b, C_out, C_total, HW,
        total_copy_elements,
    )
    
    return out


def replacement_func():
    return optimized_conv_stack_sum_cat