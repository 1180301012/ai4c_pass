import torch
import triton
import triton.language as tl

def pattern(in_0, in_2, in_3):
    # Match the sequence: in_3 += in_0; in_4 = in_3; in_4 += in_2; tmp_2 = torch.nn.functional.relu(tmp_0, inplace=True)
    # Note: We cannot use in-place ops in pattern matching, so we simulate the result
    temp = in_3 + in_0    # Regular addition (not in-place)
    temp2 = temp + in_2   # Regular addition (not in-place) 
    tmp_2 = torch.nn.functional.relu(temp2, inplace=True)
    # tmp_2 is the only observable result from this stream
    return tmp_2

def replacement_args(in_0, in_2, in_3):
    return (in_0, in_2, in_3)

@triton.jit
def fused_add_add_relu_kernel(
    x_ptr,
    y_ptr, 
    z_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load all three input tensors
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    z = tl.load(z_ptr + offsets, mask=mask, other=0.0)
    
    # Fused computation: result = relu((z + x) + y)
    temp = z + x
    out = temp + y
    out = tl.where(out > 0, out, 0.0)
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_add_add_relu(x, y, z):
    # Reshape inputs to 1D for contiguous memory access
    x_flat = x.view(-1)
    y_flat = y.view(-1)
    z_flat = z.view(-1)
    
    # Ensure all tensors have the same number of elements
    assert x_flat.numel() == y_flat.numel() == z_flat.numel()
    N = x_flat.numel()
    
    # Launch kernel with autotuning-friendly block size
    BLOCK_SIZE = 512
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x_flat)
    
    fused_add_add_relu_kernel[(num_programs,)](
        x_ptr=x_flat,
        y_ptr=y_flat,
        z_ptr=z_flat,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape back to original 4D shape [1, 128, 16, 12]
    return out.view(x.shape)

def replacement_func():
    return fused_add_add_relu