import torch
import triton
import triton.language as tl


@triton.jit
def mean_reduction_kernel(
    input_ptr, output_ptr,
    batch_size, seq_len, hidden_size,
    BLOCK_SIZE: tl.constexpr
):
    """Triton kernel for mean reduction over dim=-2 (seq_len dimension), keeping dim"""
    pid = tl.program_id(0)
    n_elements = batch_size * hidden_size
    
    if pid * BLOCK_SIZE >= n_elements:
        return
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    batch_idx = offsets // hidden_size
    hidden_idx = offsets % hidden_size
    
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Sum over seq_len dimension (dim=-2)
    for s in range(seq_len):
        in_idx = batch_idx * seq_len * hidden_size + s * hidden_size + hidden_idx
        val = tl.load(input_ptr + in_idx, mask=mask, other=0.0)
        acc += val
    
    # Divide by seq_len to get mean
    result = acc / seq_len
    
    # Store to output
    out_idx = batch_idx * hidden_size + hidden_idx
    tl.store(output_ptr + out_idx, result, mask=mask)


@torch.fx.wrap
def triton_mean_reduction(in_2):
    """
    Optimized mean reduction using Triton kernel.
    Replaces: in_2.mean(dim=-2, keepdim=True)
    Input: [B, S, C] -> Output: [B, 1, C]
    """
    B, S, C = in_2.shape
    
    # Allocate output: [B, 1, C]
    tmp_4 = torch.empty((B, 1, C), dtype=in_2.dtype, device=in_2.device)
    
    # Flatten output for kernel
    tmp_4_flat = tmp_4.view(B * C)
    
    BLOCK_SIZE = 1024
    n_elements = B * C
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    mean_reduction_kernel[(num_programs,)](
        in_2, tmp_4_flat,
        B, S, C,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return tmp_4


def pattern(in_0, in_1, in_2, in_3):
    """Match the conv2d + view + mean pattern"""
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.view(2, 256, -1)
    tmp_4 = in_2.mean(dim=-2, keepdim=True)
    return tmp_4, tmp_3


def replacement_args(in_0, in_1, in_2, in_3):
    """Extract arguments for the replacement function"""
    return (in_2,)


def replacement_func():
    """Return the replacement function"""
    return triton_mean_reduction