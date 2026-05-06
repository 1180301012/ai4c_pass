import torch
import triton
import triton.language as tl

def pattern(tmp_3, in_3):
    matmul = torch.matmul(tmp_3, in_3)
    permuted = matmul.permute(0, 2, 1, 3)
    contiguous = permuted.contiguous()
    return (contiguous,)

def replacement_args(tmp_3, in_3):
    return (tmp_3, in_3)

def triton_matmul_kernel(
    tmp3_ptr,
    in3_ptr,
    out_ptr,
    batch_size: tl.constexpr,
    heads: tl.constexpr,
    seq_len: tl.constexpr,
    dim: tl.constexpr,
    block_size: tl.constexpr,
):
    # Set up grid and block
    block_id = tl.program_id(0)
    offset = block_id * block_size
    
    # Get elements within block
    offsets = tl.arange(0, block_size, 1)
    mask = offsets < seq_len
    
    # Load data
    tmp3 = tl.load(tmp3_ptr + offset + offsets, mask=mask, other=0.0)
    in3 = tl.load(in3_ptr + offset + offsets, mask=mask, other=0.0)
    
    # Compute operation
    out = tmp3 @ in3
    
    # Store results
    tl.store(out_ptr + offset + offsets, out, mask=mask)

def kernel_wrapper(tmp_3, in_3):
    # Extract tensor dimensions
    batch, heads, seq_len, dim = tmp_3.shape
    
    # Create output tensor with correct shape [batch, seq_len, heads, dim]
    out = torch.empty(batch, seq_len, heads, dim, dtype=tmp_3.dtype, device=tmp_3.device)
    
    # Configure kernel launch
    grid = (seq_len,)
    triton_matmul_kernel[
        grid
    ](  
        tmp3_ptr=tmp_3,
        in3_ptr=in_3,
        out_ptr=out,
        batch_size=batch,
        heads=heads,
        seq_len=seq_len,
        dim=dim,
        block_size=256,
    )
    
    return out

def replacement_func():
    return kernel_wrapper