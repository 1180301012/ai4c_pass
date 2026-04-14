import torch
import triton
import triton.language as tl

def pattern(x):
    """
    Pattern matches: pad operation with (0, 1, 0, 1) padding
    This pads 1 element to the right and bottom of the tensor
    """
    return torch.nn.functional.pad(x, (0, 1, 0, 1), 'constant', None)

def replacement_args(x):
    return (x,)

@triton.jit
def optimized_padding_kernel(
    x_ptr,
    out_ptr,
    H: tl.constexpr,
    W: tl.constexpr,
    C: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get thread/block IDs
    pid = tl.program_id(0)
    # Each thread processes one spatial location (H, W)
    h = pid // W
    w = pid % W
    
    # Check bounds for original input
    if h >= H and w >= W:
        # This thread is outside original bounds but within padded area
        # Set to constant (0.0 for 'constant' padding with default 0)
        out_offset = h * (W + 1) + w
        if h == H and w < W:
            # Right padding column
            for c in range(0, C, BLOCK_SIZE):
                mask = c + tl.arange(0, BLOCK_SIZE) < C
                offsets = out_offset * C + c + tl.arange(0, BLOCK_SIZE)
                tl.store(out_ptr + offsets, 0.0, mask=mask)
        elif w == W and h < H:
            # Bottom padding row
            out_offset = h * (W + 1) * C
            for c in range(0, C, BLOCK_SIZE):
                mask = c + tl.arange(0, BLOCK_SIZE) < C
                offsets = out_offset + c + tl.arange(0, BLOCK_SIZE)
                tl.store(out_ptr + offsets, 0.0, mask=mask)
        else:
            # Both H and W are padded (bottom-right corner)
            out_offset = (h * (W + 1) + w) * C
            for c in range(0, C, BLOCK_SIZE):
                mask = c + tl.arange(0, BLOCK_SIZE) < C
                offsets = out_offset + c + tl.arange(0, BLOCK_SIZE)
                tl.store(out_ptr + offsets, 0.0, mask=mask)
    elif h < H and w < W:
        # Normal data from input
        in_offset = h * W + w
        out_offset = h * (W + 1) + w  # Account for right padding
        
        for c in range(0, C, BLOCK_SIZE):
            mask = c + tl.arange(0, BLOCK_SIZE) < C
            # Load from input
            x_val = tl.load(x_ptr + (in_offset * C + c + tl.arange(0, BLOCK_SIZE)), mask=mask, other=0.0)
            # Store to output (accounting for padding)
            tl.store(out_ptr + (out_offset * C + c + tl.arange(0, BLOCK_SIZE)), x_val, mask=mask)

@torch.fx.wrap
def optimized_padding(x):
    # Get input tensor shape [N, C, H, W]
    if x.dim() != 4:
        # For non-4D tensors or fallback, return as-is (caller handles padding)
        return x
    
    N, C, H, W = x.shape
    
    # Calculate output dimensions (H+1, W+1)
    out_H, out_W = H + 1, W + 1
    
    # Create output tensor with same dtype and device
    out = torch.empty((N, C, out_H, out_W), dtype=x.dtype, device=x.device)
    
    # Total spatial locations (H+1) * (W+1)  
    total_spatial = out_H * out_W
    
    # Get optimal block size for the channel dimension
    BLOCK_SIZE = 32  # Good for most GPUs
    num_programs = total_spatial
    
    # Handle each batch dimension separately
    for n in range(N):
        x_ptr_n = x[n].contiguous()
        out_ptr_n = out[n].contiguous()
        
        optimized_padding_kernel[(num_programs,)](
            x_ptr=x_ptr_n,
            out_ptr=out_ptr_n,
            H=H,
            W=W,
            C=C,
            N=1,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return out

def replacement_func():
    return optimized_padding