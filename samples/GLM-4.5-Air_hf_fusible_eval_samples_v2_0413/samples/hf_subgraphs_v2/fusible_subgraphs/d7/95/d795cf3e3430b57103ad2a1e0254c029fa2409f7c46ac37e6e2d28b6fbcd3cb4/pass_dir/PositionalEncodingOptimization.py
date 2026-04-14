import torch
import triton
import triton.language as tl

# Pattern that matches the exact computation
def pattern():
    """Pattern that matches the complete computation from model.py"""
    # Use default device (framework will handle placement)
    tmp_0 = torch.zeros((1, 133, 133))
    
    tmp_1 = tmp_0[(slice(None, None, None), slice(-5, None, None), slice(None, None, None))]
    tmp_2 = tmp_1.fill_(1)
    
    tmp_3 = tmp_0[(slice(None, None, None), slice(None, None, None), slice(-5, None, None))]
    tmp_4 = tmp_3.fill_(1)
    
    tmp_7 = tmp_0.reshape(1, 19, 7, 19, 7)
    tmp_8 = tmp_7.transpose(2, 3)
    tmp_9 = tmp_8.reshape(1, 361, 49)
    
    tmp_10 = tmp_9.unsqueeze(2)
    tmp_11 = tmp_9.unsqueeze(3)
    tmp_12 = tmp_10 - tmp_11
    
    tmp_13 = tmp_12 != 0
    tmp_14 = tmp_12.masked_fill(tmp_13, -1000.0)
    
    tmp_15 = tmp_12 == 0
    tmp_16 = tmp_14.masked_fill(tmp_15, 0.0)
    
    return tmp_16

def replacement_args():
    return ()

@triton.jit
def create_positional_encoding_kernel(
    out_ptr,
    H: tl.constexpr,
    W: tl.constexpr,
    block_size: tl.constexpr,
):
    # Create efficient positional encoding in Triton
    pid = tl.program_id(0)
    idx = pid * block_size + tl.arange(0, block_size)
    
    mask = idx < H * W
    
    if tl.constexpr(mask):
        h = idx // W
        w = idx % W
        
        # Create relative position encoding
        pos_h = h.unsqueeze(-1) - h.unsqueeze(-2)
        pos_w = w.unsqueeze(-1) - w.unsqueeze(-2)
        
        # Apply masking for positions beyond 5
        pos_mask = ((pos_h.abs() < 5) & (pos_w.abs() < 5)).to(tl.float32)
        
        # Fill with -1000 where positions are different
        out_vals = -1000.0 * (1 - pos_mask)
        # Fill with 0 where positions are same
        out_vals = out_vals * (tl.abs(pos_h - pos_w) > 0).to(tl.float32)
        out_vals = out_vals * 0.0
        
        tl.store(out_ptr + idx, out_vals, mask=mask)

def optimized_positional_encoding():
    H = 361  # 19 * 19
    W = 49   # 7 * 7
    out = torch.zeros(1, H, W, dtype=torch.float16, device='cuda')
    
    block_size = 1024
    total_elements = H * W
    num_programs = (total_elements + block_size - 1) // block_size
    
    create_positional_encoding_kernel[(num_programs,)](
        out_ptr=out,
        H=H,
        W=W,
        block_size=block_size,
    )
    
    return out

# Modified pattern that matches the exact computation in the graphs
def exact_pattern():
    from torch import device
    tmp_0 = torch.zeros((1, 133, 133), device=device(type='cuda', index=0))
    
    tmp_1 = tmp_0[(slice(None, None, None), slice(-5, None, None), slice(None, None, None))]
    tmp_2 = tmp_1.fill_(1)
    
    tmp_3 = tmp_0[(slice(None, None, None), slice(None, None, None), slice(-5, None, None))]
    tmp_4 = tmp_3.fill_(1)
    
    tmp_7 = tmp_0.reshape(1, 19, 7, 19, 7)
    tmp_8 = tmp_7.transpose(2, 3)
    tmp_9 = tmp_8.reshape(1, 361, 49)
    
    tmp_10 = tmp_9.unsqueeze(2)
    tmp_11 = tmp_9.unsqueeze(3)
    tmp_12 = tmp_10 - tmp_11
    
    tmp_13 = tmp_12 != 0
    tmp_14 = tmp_12.masked_fill(tmp_13, -1000.0)
    
    tmp_15 = tmp_12 == 0
    tmp_16 = tmp_14.masked_fill(tmp_15, 0.0)
    
    return tmp_16

@triton.jit
def fast_positional_encoding_kernel(
    out_ptr,
    h: tl.constexpr,
    w: tl.constexpr,
    block_size: tl.constexpr,
):
    """Fast kernel for creating relative position encoding"""
    pid = tl.program_id(0)
    idx = pid * block_size + tl.arange(0, block_size)
    mask = idx < h * w
    
    if tl.constexpr(mask):
        # Create position indices
        i = idx // w
        j = idx % w
        
        # Create relative positions
        pos_i = i.unsqueeze(-1) - i.unsqueeze(-2)
        pos_j = j.unsqueeze(-1) - j.unsqueeze(-2)
        
        # Create mask for positions within bounds
        valid_pos = (tl.abs(pos_i) >= 0) & (tl.abs(pos_j) >= 0)
        
        # Initialize output with zeros
        out_vals = tl.zeros_like(pos_i, dtype=tl.float32)
        
        # Fill positions with -1000 where different
        different_pos = (pos_i != 0) | (pos_j != 0)
        out_vals = tl.where(different_pos & valid_pos, -1000.0, out_vals)
        
        tl.store(out_ptr + idx, out_vals, mask=mask)

@torch.fx.wrap
def fast_positional_encoding():
    """Optimized implementation of the positional encoding computation"""
    # Target shape: 1, 361, 49
    h, w = 361, 49
    out = torch.empty(1, h, w, dtype=torch.float16, device='cuda')
    
    # Launch Triton kernel
    block_size = 1024
    total_elements = h * w
    num_programs = (total_elements + block_size - 1) // block_size
    
    fast_positional_encoding_kernel[(num_programs,)](
        out_ptr=out,
        h=h,
        w=w,
        block_size=block_size,
    )
    
    return out

# Pattern that matches the exact computation
def pattern():
    """Pattern that matches the complete computation from model.py"""
    # Use default device (framework will handle placement)
    tmp_0 = torch.zeros((1, 133, 133))
    
    tmp_1 = tmp_0[(slice(None, None, None), slice(-5, None, None), slice(None, None, None))]
    tmp_2 = tmp_1.fill_(1)
    
    tmp_3 = tmp_0[(slice(None, None, None), slice(None, None, None), slice(-5, None, None))]
    tmp_4 = tmp_3.fill_(1)
    
    tmp_7 = tmp_0.reshape(1, 19, 7, 19, 7)
    tmp_8 = tmp_7.transpose(2, 3)
    tmp_9 = tmp_8.reshape(1, 361, 49)
    
    tmp_10 = tmp_9.unsqueeze(2)
    tmp_11 = tmp_9.unsqueeze(3)
    tmp_12 = tmp_10 - tmp_11
    
    tmp_13 = tmp_12 != 0
    tmp_14 = tmp_12.masked_fill(tmp_13, -1000.0)
    
    tmp_15 = tmp_12 == 0
    tmp_16 = tmp_14.masked_fill(tmp_15, 0.0)
    
    return tmp_16

def replacement_args():
    return ()

def replacement_func():
    return fast_positional_encoding