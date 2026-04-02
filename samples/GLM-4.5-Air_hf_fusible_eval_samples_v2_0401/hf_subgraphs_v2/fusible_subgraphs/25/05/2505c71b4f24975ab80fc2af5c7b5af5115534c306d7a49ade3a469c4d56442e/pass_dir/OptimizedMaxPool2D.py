import torch
import triton
import triton.language as tl

def pattern(x):
    # This matches: max_pool2d with kernel_size=2, stride=1, padding=0, ceil_mode=True
    out = torch.nn.functional.max_pool2d(x, 2, 1, 0, 1, ceil_mode=True, return_indices=False)
    return out

def replacement_args(x):
    return (x,)

@triton.jit
def optimized_max_pool2d_kernel(
    x_ptr,
    out_ptr,
    N,
    C,
    H,
    W,
    KH: tl.constexpr,
    KW: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = N * C * H * W
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Reshape offsets to 4D coordinates
    w = offsets % W
    h = (offsets // W) % H
    c = (offsets // (W * H)) % C
    n = offsets // (W * H * C)
    
    # Apply padding (0 in this case)
    padded_h = h + PH
    padded_w = w + PW
    
    # Calculate output dimensions according to ceil_mode=True
    # Out_H = floor((H + 2*PH - KH) / SH + 1) when ceil_mode=True
    out_H = (H + 2*PH - KH + SH - 1) // SH + 1
    out_W = (W + 2*PW - KW + SW - 1) // SW + 1
    
    # For each output pixel, check the corresponding window in input
    values = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    
    # Kernel window: KH=2, KW=2, SH=1, SW=1
    for kh in range(KH):
        for kw in range(KW):
            # Calculate input coordinates
            in_h = padded_h * SH + kh
            in_w = padded_w * SW + kw
            
            # Check bounds
            in_h_mask = (in_h < H) & (in_h >= 0)
            in_w_mask = (in_w < W) & (in_w >= 0)
            valid_mask = in_h_mask & in_w_mask & mask
            
            # Calculate input index
            in_idx = n * (W * H * C) + c * (W * H) + in_h * W + in_w
            
            # Load input value and update max
            x_val = tl.load(x_ptr + in_idx, mask=valid_mask, other=float('-inf'))
            values = tl.maximum(values, x_val)
    
    # Store output
    out_idx = n * (out_W * out_H * C) + c * (out_W * out_H) + h * out_W + w
    tl.store(out_ptr + out_idx, values, mask=mask)

@torch.fx.wrap
def optimized_max_pool2d(x):
    N, C, H, W = x.shape
    KH, KW = 2, 2  # kernel_size=2
    SH, SW = 1, 1  # stride=1
    PH, PW = 0, 0  # padding=0
    
    # Calculate output dimensions with ceil_mode=True
    out_H = (H + 2*PH - KH + SH - 1) // SH + 1
    out_W = (W + 2*PW - KW + SW - 1) // SW + 1
    out = torch.empty((N, C, out_H, out_W), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 1024  # Optimal block size
    total_elements = N * C * out_H * out_W
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_max_pool2d_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        N=N,
        C=C,
        H=H,
        W=W,
        KH=KH,
        KW=KW,
        SH=SH,
        SW=SW,
        PH=PH,
        PW=PW,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_max_pool2d