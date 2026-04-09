import torch
import triton
import triton.language as tl

def pattern(x):
    return torch.nn.functional.pad(x, (0, 1, 0, 1), 'constant', None)

def replacement_args(x):
    return (x,)

@triton.jit
def optimized_pad_kernel(
    x_ptr,
    out_ptr,
    original_height,
    original_width,
    pad_bottom,
    pad_right,
    batch_size,
    channels,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size * channels * original_height * original_width
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x, mask=mask)

@torch.fx.wrap
def optimized_pad(x):
    pad_right = 1
    pad_bottom = 1
    
    batch, channels, height, width = x.shape
    output_height = height + pad_bottom
    output_width = width + pad_right
    output_shape = (batch, channels, output_height, output_width)
    
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Copy original content to output using Triton kernel
    if x.numel() > 0:
        # Use dynamic block sizes based on tensor size
        if x.numel() >= 1024 * 1024:  # Large tensors
            BLOCK_SIZE = 2048
        elif x.numel() >= 256 * 1024:  # Medium tensors
            BLOCK_SIZE = 1024
        else:  # Small tensors
            BLOCK_SIZE = 512
            
        num_programs = (x.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        optimized_pad_kernel[(num_programs,)](
            x_ptr=x,
            out_ptr=out,
            original_height=height,
            original_width=width,
            pad_bottom=pad_bottom,
            pad_right=pad_right,
            batch_size=batch,
            channels=channels,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    # Set padding values to 0 (optimized for small padding regions)
    if pad_bottom > 0:
        out[:, :, height:, :] = 0
    if pad_right > 0:
        out[:, :, :, width:] = 0
        
    return out

def replacement_func():
    return optimized_pad