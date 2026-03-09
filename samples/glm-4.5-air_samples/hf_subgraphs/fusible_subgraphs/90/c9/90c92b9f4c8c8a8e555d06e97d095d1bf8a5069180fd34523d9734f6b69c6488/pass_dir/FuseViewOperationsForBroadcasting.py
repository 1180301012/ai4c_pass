import torch
import triton
import triton.language as tl

def pattern(softmax_output, input_0):
    # Match the sequence of view operations that can be fused
    tmp_1 = softmax_output.reshape(-1)
    tmp_2 = tmp_1.view(-1, 1, 1)
    tmp_3 = tmp_2.view(2, -1, 1, 1)
    return tmp_3

def replacement_args(softmax_output, input_0):
    return (softmax_output, input_0)

@triton.jit
def fused_view_broadcast_kernel(
    softmax_ptr,
    input_0_ptr,
    out_ptr,
    batch_size: tl.constexpr,
    hidden_dim: tl.constexpr,
    spatial_h: tl.constexpr,
    spatial_w: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a spatial location
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    mask = offset + tl.arange(0, BLOCK_SIZE) < (spatial_h * spatial_w)
    
    # Calculate indices for each dimension
    spatial_idx = offset + tl.arange(0, BLOCK_SIZE)
    h = spatial_idx // spatial_w
    w = spatial_idx % spatial_w
    
    # Broadcast the softmax output: [batch, 2, 4, 1, 1] -> [batch, 2, hidden_dim, spatial_h, spatial_w]
    # We need to tile the 4x1x1 pattern to match the hidden_dim = 128
    for k in range(0, hidden_dim, 4):
        k_idx = tl.arange(k, min(k + 4, hidden_dim))
        
        # Load softmax output (reshaped to [batch, 2, 4, 1, 1])
        batch_idx = tl.program_id(1)
        channel_idx = tl.program_id(2)
        hidden_idx = k_idx % 4  # Map hidden_dim to the 4-element broadcast pattern
        
        softmax_val = tl.load(softmax_ptr + batch_idx * (2 * 4) + channel_idx * 4 + hidden_idx)
        
        # Broadcast across spatial dimensions
        for i in range(0, spatial_h, BLOCK_SIZE // 8):
            for j in range(0, spatial_w, 1):
                h_idx = i + tl.arange(0, min(BLOCK_SIZE // 8, spatial_h - i))
                w_idx = j
                
                spatial_mask = (h_idx < spatial_h) & (w_idx < spatial_w)
                
                # Compute output indices
                out_idx = (batch_idx * (2 * hidden_dim * spatial_h * spatial_w) + 
                          channel_idx * (hidden_dim * spatial_h * spatial_w) +
                          k_idx * (spatial_h * spatial_w) +
                          h_idx * spatial_w + w_idx * tl.broadcast_to(1, len(h_idx)))
                
                # Load input_0
                input_val = tl.load(input_0_ptr + out_idx, mask=spatial_mask, other=0.0)
                
                # Multiply
                result = softmax_val * input_val
                
                # Store
                tl.store(out_ptr + out_idx, result, mask=spatial_mask)

@torch.fx.wrap
def fused_view_broadcast(softmax_output, input_0):
    batch_size, two_dim, one_dim, hidden_dim = softmax_output.shape
    spatial_h = input_0.shape[3]
    spatial_w = input_0.shape[4]
    
    # Output shape should be [batch_size, 2, hidden_dim, spatial_h, spatial_w]
    out_shape = (batch_size, 2, hidden_dim, spatial_h, spatial_w)
    out = torch.zeros(out_shape, dtype=softmax_output.dtype, device=softmax_output.device)
    
    # Launch kernel with proper grid configuration
    spatial_elements = spatial_h * spatial_w
    BLOCK_SIZE = 1024
    num_spatial_programs = (spatial_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_batch_programs = batch_size
    num_channel_programs = 2
    
    fused_view_broadcast_kernel[(num_spatial_programs, num_batch_programs, num_channel_programs)](
        softmax_ptr=softmax_output,
        input_0_ptr=input_0,
        out_ptr=out,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        spatial_h=spatial_h,
        spatial_w=spatial_w,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_view_broadcast