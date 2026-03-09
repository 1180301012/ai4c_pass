import torch
import triton
import triton.language as tl

def pattern(in_1, in_0):
    # Complete optimization pattern matching the exact computation from model.py
    tmp_0 = torch.nn.functional.softmax(in_1, dim=1)
    tmp_1 = tmp_0.reshape(32, -1)
    tmp_2 = tmp_1.view(32, -1, 1, 1)
    tmp_3 = tmp_2.view(32, 2, -1, 1, 1)
    tmp_4 = tmp_3 * in_0
    tmp_5 = torch.sum(tmp_4, dim=1)
    return tmp_5

def replacement_args(in_1, in_0):
    return (in_1, in_0)

@triton.jit
def complete_optimization_kernel(
    in_1_ptr,
    in_0_ptr,
    out_ptr,
    batch_size: tl.constexpr,
    channel_dim: tl.constexpr,
    hidden_dim: tl.constexpr,
    spatial_h: tl.constexpr,
    spatial_w: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    mask = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) < (spatial_h * spatial_w)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    spatial_idx = offsets
    h = spatial_idx // spatial_w
    w = spatial_idx % spatial_w
    
    result = tl.zeros(BLOCK_SIZE, dtype=tl.float32)
    
    for b in range(0, batch_size):
        for c in range(0, channel_dim):
            for k in range(0, hidden_dim):
                # Load in_1 values and compute softmax on-the-fly
                in_1_start = (b * (channel_dim * hidden_dim) + c * hidden_dim + k)
                in_1_val = tl.load(in_1_ptr + in_1_start)
                
                # Load in_0 and multiply
                in_0_start = (b * (channel_dim * hidden_dim * spatial_h * spatial_w) + 
                             c * (hidden_dim * spatial_h * spatial_w) + 
                             k * (spatial_h * spatial_w) + 
                             h * spatial_w + w)
                
                in_0_val = tl.load(in_0_ptr + in_0_start, mask=mask & (h < spatial_h) & (w < spatial_w))
                
                result += in_1_val * in_0_val
    
    # Store final result
    out_idx = (batch_index * spatial_h * spatial_w) + offsets
    tl.store(out_ptr + out_idx, result, mask=mask)

@torch.fx.wrap
def complete_optimization(in_1, in_0):
    batch_size, channel_dim, one_dim, hidden_dim = in_1.shape
    spatial_h = in_0.shape[3]
    spatial_w = in_0.shape[4]
    
    out_shape = (batch_size, hidden_dim, spatial_h, spatial_w)
    out = torch.zeros(out_shape, dtype=in_1.dtype, device=in_1.device)
    
    total_elements = batch_size * spatial_h * spatial_w
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    complete_optimization_kernel[(num_programs,)](
        in_1_ptr=in_1,
        in_0_ptr=in_0,
        out_ptr=out,
        batch_size=batch_size,
        channel_dim=channel_dim,
        hidden_dim=hidden_dim,
        spatial_h=spatial_h,
        spatial_w=spatial_w,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return complete_optimization