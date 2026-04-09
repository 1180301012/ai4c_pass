import torch
import triton
import triton.language as tl

@triton.jit
def optimized_concat_channels_kernel(
    tensor1_ptr,
    tensor2_ptr,
    out_ptr,
    batch_size,
    channels1,
    height,
    width,
    channels2,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate total elements
    total_elements = batch_size * (channels1 + channels2) * height * width
    linear_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create mask
    mask = linear_idx < total_elements
    
    # Calculate indices
    h = (linear_idx // (width * (channels1 + channels2))) % height
    w = (linear_idx // (channels1 + channels2)) % width
    c = (linear_idx // (width * height)) % (channels1 + channels2)
    b = linear_idx // (width * height * (channels1 + channels2))
    
    result = tl.zeros(BLOCK_SIZE, dtype=tl.float32)
    
    for i in range(BLOCK_SIZE):
        if mask[i]:
            c_idx = c[i]
            linear_idx_4d = b.item() * width * height * (channels1 + channels2) + h[i].item() * width * (channels1 + channels2) + w[i].item() * (channels1 + channels2) + c_idx
            
            if c_idx < channels1:
                # From first tensor
                src_linear_idx = b.item() * width * height * channels1 + h[i].item() * width * channels1 + w[i].item() * channels1 + c_idx
                val = tl.load(tensor1_ptr + src_linear_idx, other=0.0)
            else:
                # From second tensor
                src_linear_idx = b.item() * width * height * channels2 + h[i].item() * width * channels2 + w[i].item() * channels2 + (c_idx - channels1)
                val = tl.load(tensor2_ptr + src_linear_idx, other=0.0)
            
            result[i] = val
    
    # Store result
    tl.store(out_ptr + linear_idx, result, mask=mask)

@torch.fx.wrap
def optimized_concat_channels(tensor1, tensor2, dim=1):
    assert dim == 1, "This optimization only supports concatenation along channel dimension (dim=1)"
    
    batch_size, channels1, height, width = tensor1.shape
    _, channels2, _, _ = tensor2.shape
    
    output = torch.empty(batch_size, channels1 + channels2, height, width, dtype=tensor1.dtype, device=tensor1.device)
    
    # Optimized block size for large tensors
    BLOCK_SIZE = 1024
    total_elements = batch_size * (channels1 + channels2) * height * width
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_concat_channels_kernel[(num_programs,)](
        tensor1_ptr=tensor1,
        tensor2_ptr=tensor2,
        out_ptr=output,
        batch_size=batch_size,
        channels1=channels1,
        height=height,
        width=width,
        channels2=channels2,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def pattern(t1, t2):
    """Pattern: tensor concatenation along channel dimension (dim=1)"""
    return torch.cat([t1, t2], dim=1)

def replacement_args(t1, t2):
    return (t1, t2)

def replacement_func():
    return optimized_concat_channels