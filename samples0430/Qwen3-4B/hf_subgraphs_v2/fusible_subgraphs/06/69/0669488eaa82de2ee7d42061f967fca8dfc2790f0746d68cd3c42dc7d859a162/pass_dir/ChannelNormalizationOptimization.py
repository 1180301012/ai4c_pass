import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    scale = 0.14433756729740643
    relu = torch.nn.functional.relu(in_1, inplace=True)
    flattened = torch.flatten(relu, 2)
    norm = torch.norm(flattened, dim=-1, keepdim=True)
    scaled = norm * scale
    clamped = scaled.clamp(min=1e-05)
    divided = flattened / clamped
    result = divided * in_0
    return result

def replacement_args(in_0, in_1):
    return in_0, in_1

@triton.jit
def optimized_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    in_1_shape,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Placeholder kernel implementation
    i = tl.program_id(0)
    pos = i * BLOCK_SIZE
    idx = tl.arange(0, BLOCK_SIZE)
    mask = (pos + idx) < tl.arange(0, in_1_shape[0]) * tl.arange(0, in_1_shape[1]) * tl.arange(0, in_1_shape[2])
    
    in_1 = tl.load(in_1_ptr + pos + idx, mask=mask, other=0.0)
    relu = in_1.clamp(min=0.0)
    
    # Compute flattened tensor (simplified for illustration)
    flattened = relu
    
    # Compute per-channel norm (simplified approximation)
    norm = tl.sqrt(tl.sum(flattened ** 2, dim=1))
    
    scaled = norm * scale
    clamped = scaled.clamp(min=1e-5)
    divided = flattened / clamped
    
    result = divided * tl.load(in_0_ptr + pos + idx, mask=mask, other=0.0)
    
    tl.store(out_ptr + pos + idx, result, mask=mask)

@torch.fx.wrap
def optimized_function(in_0, in_1):
    scale = 0.14433756729740643
    out = torch.empty_like(in_0)
    
    # Launch kernel with grid configuration
    optimized_kernel[
        tl.cintr(1),
    ](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        in_1_shape=in_1.shape,
        scale=scale,
        BLOCK_SIZE=1024,
    )
    
    return out

def replacement_func():
    return optimized_function