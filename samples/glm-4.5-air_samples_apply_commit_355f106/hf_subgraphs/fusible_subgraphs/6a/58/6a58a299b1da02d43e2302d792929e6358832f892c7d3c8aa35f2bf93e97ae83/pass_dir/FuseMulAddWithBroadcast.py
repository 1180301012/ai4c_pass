import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    # Pattern matches: computation = in_2 * in_1 + in_0
    tmp_1 = in_2 * in_1
    tmp_2 = tmp_1 + in_0
    return tmp_2

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def fused_muladd_kernel(
    out_ptr,
    a_ptr,  # in_0
    b_ptr,  # in_1 
    c_ptr,  # in_2
    nelements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and compute memory offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < nelements
    
    # Load inputs - broadcasting should be handled by PyTorch before calling kernel
    # The tensors should already be broadcasted to the same shape
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    
    # Fused multiply-add: optimized computation order
    # Compute result where mask is True, zero elsewhere
    result = tl.where(mask, a + b * c, 0.0)
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_muladd(in_0, in_1, in_2):
    # Determine output shape based on broadcasting rules
    # Calculate broadcasted shape manually
    input_shapes = [in_0.shape, in_1.shape, in_2.shape]
    max_dims = max(len(s) for s in input_shapes)
    padded_shapes = []
    for shape in input_shapes:
        padded_shape = (1,) * (max_dims - len(shape)) + shape
        padded_shapes.append(padded_shape)
    
    output_shape = []
    for i in range(max_dims):
        dim_size = 1
        for shape in padded_shapes:
            if shape[i] != 1 and dim_size != 1:
                dim_size = max(dim_size, shape[i])
            else:
                dim_size = max(dim_size, shape[i])
        output_shape.append(dim_size)
    
    output_shape = tuple(output_shape)
    out = torch.empty(output_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Calculate total elements and launch kernel
    nelements = 1
    for dim in output_shape:
        nelements *= dim
    
    # Adaptive block size based on input size
    if nelements < 10000:
        BLOCK_SIZE = 256
    elif nelements < 100000:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 1024
    
    num_programs = (nelements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_muladd_kernel[(num_programs,)](
        out_ptr=out,
        a_ptr=in_0,
        b_ptr=in_1,
        c_ptr=in_2,
        nelements=nelements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_muladd