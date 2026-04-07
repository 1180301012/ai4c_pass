import torch
import triton
import triton.language as tl

@triton.jit
def fused_arange_bool_kernel(range_ptr, bool_ptr, input_ptr, seq_len, BLOCK_SIZE: tl.constexpr):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < seq_len
    
    # Generate range values [0, 1, 2, ...]
    range_vals = idx
    tl.store(range_ptr + idx, range_vals, mask=mask)
    
    # Convert input tensor to bool
    if input_ptr is not None:
        input_vals = tl.load(input_ptr + idx, mask=mask, other=0)
        bool_vals = input_vals != 0
        tl.store(bool_ptr + idx, bool_vals, mask=mask)

@torch.fx.wrap
def fused_arange_bool(seq_len, input_tensor=None):
    # Determine output shapes
    if input_tensor is not None:
        range_output = torch.empty(seq_len, dtype=torch.int32, device=input_tensor.device)
        bool_output = torch.empty(input_tensor.shape, dtype=torch.bool, device=input_tensor.device)
    else:
        range_output = torch.empty(seq_len, dtype=torch.int32, device='cuda:0')
        bool_output = None
    
    # Launch kernel
    BLOCK_SIZE = 1024
    num_programs = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_arange_bool_kernel[(num_programs,)](
        range_ptr=range_output,
        bool_ptr=bool_output if input_tensor is not None else None,
        input_ptr=input_tensor,
        seq_len=seq_len,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return range_output, bool_output

def pattern(in_0):
    from torch import device
    tmp_1 = torch.arange(0, 128, device=device(type='cuda', index=0))
    tmp_2 = in_0.to(device=device(type='cuda', index=0), dtype=torch.bool)
    return tmp_1, tmp_2

def replacement_args(in_0):
    return (in_0,)

def replacement_func():
    return lambda in_0: fused_arange_bool(128, in_0)