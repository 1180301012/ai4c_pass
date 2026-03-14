import torch
import triton
import triton.language as tl

# Pattern for return order (tmp_6, tmp_4)
def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_3 + in_2
    tmp_3 = tmp_2 * in_1
    tmp_4 = tmp_3 + in_0
    tmp_5 = torch.tensor(1000)
    tmp_6 = tmp_4[slice(None, None, None), 0]
    return (tmp_6, tmp_4)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_add_mul_add_slice_kernel_b(
    in_0_ptr, in_1_ptr, in_2_ptr, in_3_ptr,
    out_ptr, slice_out_ptr,
    seq_len, hidden_dim,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Compute hidden dimension index for broadcasting
    hidden_idx = offsets % hidden_dim
    
    # Load values from 3D tensors
    in_2_vals = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)
    in_3_vals = tl.load(in_3_ptr + offsets, mask=mask, other=0.0)
    
    # Load values from 1D tensors (broadcast)
    in_0_vals = tl.load(in_0_ptr + hidden_idx, mask=mask, other=0.0)
    in_1_vals = tl.load(in_1_ptr + hidden_idx, mask=mask, other=0.0)
    
    # Compute: (in_3 + in_2) * in_1 + in_0
    result = (in_3_vals + in_2_vals) * in_1_vals + in_0_vals
    
    # Store main output
    tl.store(out_ptr + offsets, result, mask=mask)
    
    # For slice output, determine which elements are at seq_idx=0
    seq_hidden = seq_len * hidden_dim
    batch_idx = offsets // seq_hidden
    remainder = offsets % seq_hidden
    seq_idx = remainder // hidden_dim
    
    # Only store elements where seq_idx == 0
    slice_mask = mask & (seq_idx == 0)
    slice_offsets = batch_idx * hidden_dim + hidden_idx
    tl.store(slice_out_ptr + slice_offsets, result, mask=slice_mask)

@torch.fx.wrap
def fused_add_mul_add_slice_reversed(in_0, in_1, in_2, in_3):
    batch_size, seq_len, hidden_dim = in_2.shape
    n_elements = in_2.numel()
    
    out = torch.empty_like(in_2)
    slice_out = torch.empty((batch_size, hidden_dim), device=in_2.device, dtype=in_2.dtype)
    
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    
    fused_add_mul_add_slice_kernel_b[grid](
        in_0, in_1, in_2, in_3,
        out, slice_out,
        seq_len, hidden_dim,
        n_elements,
    )
    
    return slice_out, out

def replacement_func():
    return fused_add_mul_add_slice_reversed