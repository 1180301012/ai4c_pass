import torch
import triton
import triton.language as tl

# Quick GELU approximation: x * sigmoid(1.702 * x)
# This fuses: scale + sigmoid + multiply + dropout(p=0)

# Optimized kernel with larger blocks for better GPU utilization
@triton.jit
def quick_gelu_kernel(
    input_ptr,
    output_ptr,
    seq_len: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused Quick GELU kernel for 3D tensors [batch, seq, hidden]"""
    pid = tl.program_id(0)
    
    num_hidden_blocks = hidden_size // BLOCK_SIZE
    seq_idx = pid // num_hidden_blocks
    block_idx = pid % num_hidden_blocks
    
    if seq_idx >= seq_len:
        return
    
    base_offset = seq_idx * hidden_size
    hidden_offset = block_idx * BLOCK_SIZE
    
    offsets = hidden_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size
    
    x = tl.load(input_ptr + base_offset + offsets, mask=mask, other=0.0)
    scaled = 1.702 * x
    sigmoid_val = tl.sigmoid(scaled)
    out = x * sigmoid_val
    
    tl.store(output_ptr + base_offset + offsets, out, mask=mask)


@torch.fx.wrap
def quick_gelu_fused(x):
    """Fused Quick GELU optimized for 3D tensors with larger blocks"""
    batch, seq_len, hidden = x.shape
    
    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)
    
    # Use larger BLOCK_SIZE for better GPU occupancy
    # 3072 = 128 * 24
    BLOCK_SIZE = 128
    num_hidden_blocks = hidden // BLOCK_SIZE
    
    grid = (batch * num_hidden_blocks,)
    
    quick_gelu_kernel[grid](
        x_contig,
        out,
        seq_len,
        hidden,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def pattern(in_0):
    """
    Match the Quick GELU pattern with dropout:
    tmp_0 = 1.702 * in_0
    tmp_1 = torch.sigmoid(tmp_0)
    tmp_2 = in_0 * tmp_1
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.0, False, False)
    return tmp_3
    """
    tmp_0 = 1.702 * in_0
    tmp_1 = torch.sigmoid(tmp_0)
    tmp_2 = in_0 * tmp_1
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.0, False, False)
    return tmp_3


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return quick_gelu_fused