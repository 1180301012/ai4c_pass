import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Pattern to match: cat + view + layer_norm (handles any hidden size)
    """
    tmp_0 = in_0  # bias
    tmp_1 = in_1  # weight
    tmp_2 = torch.cat([in_2, in_3, in_4, in_5], -1)
    tmp_3 = tmp_2.view(1, -1, tmp_1.shape[0])  # Use weight shape to determine hidden_size
    tmp_4 = torch.nn.functional.layer_norm(tmp_3, (tmp_1.shape[0],), tmp_1, tmp_0, 1e-05)
    return tmp_4


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['quarter_size'],
)
@triton.jit
def fused_cat_view_layernorm_kernel(
    in2_ptr, in3_ptr, in4_ptr, in5_ptr,
    weight_ptr, bias_ptr, out_ptr,
    seq_len, hidden_size, quarter_size,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized fused kernel for cat + view + layer_norm
    Each program processes one sequence position (row) across all hidden dimensions
    """
    row_idx = tl.program_id(0)
    
    if row_idx >= seq_len:
        return
    
    # Calculate base offset for this row in the 4D inputs
    base_offset = row_idx * quarter_size
    
    # First pass: compute mean using all 4 quarters
    mean = 0.0
    
    # Load and sum from all 4 input tensors sequentially
    for i in range(4):
        for block_start in range(0, quarter_size, BLOCK_SIZE):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < quarter_size
            
            if i == 0:
                val = tl.load(in2_ptr + base_offset + offsets, mask=mask, other=0.0)
            elif i == 1:
                val = tl.load(in3_ptr + base_offset + offsets, mask=mask, other=0.0)
            elif i == 2:
                val = tl.load(in4_ptr + base_offset + offsets, mask=mask, other=0.0)
            else:
                val = tl.load(in5_ptr + base_offset + offsets, mask=mask, other=0.0)
            
            mean += tl.sum(tl.where(mask, val, 0.0))
    
    mean = mean / hidden_size
    
    # Second pass: compute variance
    variance = 0.0
    
    for i in range(4):
        for block_start in range(0, quarter_size, BLOCK_SIZE):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < quarter_size
            
            if i == 0:
                val = tl.load(in2_ptr + base_offset + offsets, mask=mask, other=0.0)
            elif i == 1:
                val = tl.load(in3_ptr + base_offset + offsets, mask=mask, other=0.0)
            elif i == 2:
                val = tl.load(in4_ptr + base_offset + offsets, mask=mask, other=0.0)
            else:
                val = tl.load(in5_ptr + base_offset + offsets, mask=mask, other=0.0)
            
            diff = val - mean
            variance += tl.sum(tl.where(mask, diff * diff, 0.0))
    
    variance = variance / hidden_size
    rstd = 1.0 / tl.sqrt(variance + eps)
    
    # Third pass: normalize and write output
    out_base = row_idx * hidden_size
    
    for i in range(4):
        weight_bias_offset = i * quarter_size
        
        for block_start in range(0, quarter_size, BLOCK_SIZE):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < quarter_size
            
            if i == 0:
                val = tl.load(in2_ptr + base_offset + offsets, mask=mask, other=0.0)
            elif i == 1:
                val = tl.load(in3_ptr + base_offset + offsets, mask=mask, other=0.0)
            elif i == 2:
                val = tl.load(in4_ptr + base_offset + offsets, mask=mask, other=0.0)
            else:
                val = tl.load(in5_ptr + base_offset + offsets, mask=mask, other=0.0)
            
            # Load weight and bias for this segment
            weight = tl.load(weight_ptr + weight_bias_offset + offsets, mask=mask, other=1.0)
            bias_val = tl.load(bias_ptr + weight_bias_offset + offsets, mask=mask, other=0.0)
            
            # Normalize and apply affine transformation
            normalized = (val - mean) * rstd
            output = normalized * weight + bias_val
            
            # Write to output
            out_offset = out_base + weight_bias_offset + offsets
            tl.store(out_ptr + out_offset, output, mask=mask)


@torch.fx.wrap
def fused_cat_view_layernorm(bias, weight, in_2, in_3, in_4, in_5):
    """
    Wrapper function for the fused cat+view+layernorm kernel
    Handles any hidden size dynamically
    """
    batch, h, w, quarter_size = in_2.shape
    seq_len = h * w
    hidden_size = weight.shape[0]
    
    # Allocate output
    out = torch.empty((1, seq_len, hidden_size), device=in_2.device, dtype=in_2.dtype)
    
    # Launch kernel with one program per sequence position
    grid = (seq_len,)
    
    fused_cat_view_layernorm_kernel[grid](
        in_2, in_3, in_4, in_5,
        weight, bias, out,
        seq_len, hidden_size, quarter_size,
        eps=1e-05,
    )
    
    return out


def replacement_func():
    return fused_cat_view_layernorm