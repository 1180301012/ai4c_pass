import torch
import triton
import triton.language as tl

# Pattern to match: Conv2d + Sigmoid + Mul + Cat + Cat + ChannelShuffle chains + Chunks
def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    # Conv2d with bias
    conv2d = torch.conv2d(in_6, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    # Sigmoid activation
    tmp_3 = torch.sigmoid(conv2d)
    # Element-wise multiplication
    tmp_4 = in_5 * tmp_3
    # First concatenation (channel concat)
    tmp_5 = torch.cat([in_2, in_4], dim=1)
    # Second concatenation (channel concat)
    tmp_6 = torch.cat([in_3, tmp_4], dim=1)
    # First channel shuffle path
    tmp_7 = tmp_5.view(in_2.shape[0], 2, 20, 64, 48)
    tmp_8 = torch.transpose(tmp_7, 1, 2)
    tmp_9 = tmp_8.contiguous()
    tmp_10 = tmp_9.view(in_2.shape[0], 40, 64, 48)
    # Second channel shuffle path
    tmp_11 = tmp_6.view(in_3.shape[0], 2, 40, 32, 24)
    tmp_12 = torch.transpose(tmp_11, 1, 2)
    tmp_13 = tmp_12.contiguous()
    tmp_14 = tmp_13.view(in_3.shape[0], 80, 32, 24)
    # Chunk operations
    chunk = tmp_10.chunk(2, dim=1)
    tmp_16 = chunk[0]
    tmp_17 = chunk[1]
    chunk_1 = tmp_14.chunk(2, dim=1)
    tmp_19 = chunk_1[0]
    tmp_20 = chunk_1[1]
    return (tmp_16, tmp_19, tmp_17, tmp_20)

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6)


# ============================================================================
# Triton Kernels for all operations
# ============================================================================

# Kernel 1: 1x1 Conv2d with bias
@triton.jit
def conv1x1_bias_kernel(in_ptr, weight_ptr, bias_ptr, out_ptr,
                         B: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr,
                         BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_elements = B * C_out
    offset = pid * BLOCK_SIZE
    mask = offset + tl.arange(0, BLOCK_SIZE) < num_elements
    
    # Compute b, c indices
    b_idx = (offset + tl.arange(0, BLOCK_SIZE)) // C_out
    c_idx = (offset + tl.arange(0, BLOCK_SIZE)) % C_out
    
    # Compute conv: out[b, c, 0, 0] = bias[c] + sum(ic) in[b, ic, 0, 0] * weight[c, ic, 0, 0]
    conv_val = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    
    for ic in range(C_in):
        in_val = tl.load(in_ptr + b_idx * C_in + ic, mask=mask, other=0.0)
        weight_val = tl.load(weight_ptr + c_idx * C_in + ic, mask=mask, other=0.0)
        conv_val = conv_val + in_val * weight_val
    
    tl.store(out_ptr + b_idx * C_out + c_idx, conv_val, mask=mask)


# Kernel 2: Sigmoid
@triton.jit
def sigmoid_kernel(in_ptr, out_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    mask = offset + tl.arange(0, BLOCK_SIZE) < N
    
    x = tl.load(in_ptr + offset, mask=mask, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x))
    
    tl.store(out_ptr + offset, y, mask=mask)


# Kernel 3: Broadcast multiply
@triton.jit
def broadcast_mul_kernel(in1_ptr, in2_ptr, out_ptr,
                         B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
                         BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    batch_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    w_idx = tl.program_id(2)
    
    offs_h = h_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_w = w_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    mask_h = offs_h < H
    mask_w = offs_w < W
    
    # in1 is [B, C, H, W], in2 is [B, C, 1, 1] (broadcast)
    for c in range(C):
        in1_offset = batch_idx * C * H * W + c * H * W + offs_h[:, None] * W + offs_w[None, :]
        in2_offset = batch_idx * C + c
        
        in1_val = tl.load(in1_ptr + in1_offset, mask=mask_h[:, None] & mask_w[None, :], other=0.0)
        in2_val = tl.load(in2_ptr + in2_offset, mask=mask_h[:, None] & mask_w[None, :], other=0.0)
        
        out_val = in1_val * in2_val
        
        out_offset = batch_idx * C * H * W + c * H * W + offs_h[:, None] * W + offs_w[None, :]
        tl.store(out_ptr + out_offset, out_val, mask=mask_h[:, None] & mask_w[None, :])


# Kernel 4: Channel concatenation (cat along dim=1)
@triton.jit
def concat_channel_kernel(in1_ptr, in2_ptr, out_ptr,
                          B: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr,
                          H: tl.constexpr, W: tl.constexpr,
                          BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    batch_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    w_idx = tl.program_id(2)
    
    offs_h = h_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_w = w_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    mask_h = offs_h < H
    mask_w = offs_w < W
    
    # Copy first half
    for c in range(C1):
        in1_offset = batch_idx * C1 * H * W + c * H * W + offs_h[:, None] * W + offs_w[None, :]
        out_offset = batch_idx * (C1 + C2) * H * W + c * H * W + offs_h[:, None] * W + offs_w[None, :]
        
        val = tl.load(in1_ptr + in1_offset, mask=mask_h[:, None] & mask_w[None, :], other=0.0)
        tl.store(out_ptr + out_offset, val, mask=mask_h[:, None] & mask_w[None, :])
    
    # Copy second half
    for c in range(C2):
        in2_offset = batch_idx * C2 * H * W + c * H * W + offs_h[:, None] * W + offs_w[None, :]
        out_offset = batch_idx * (C1 + C2) * H * W + (C1 + c) * H * W + offs_h[:, None] * W + offs_w[None, :]
        
        val = tl.load(in2_ptr + in2_offset, mask=mask_h[:, None] & mask_w[None, :], other=0.0)
        tl.store(out_ptr + out_offset, val, mask=mask_h[:, None] & mask_w[None, :])


# Kernel 5: Channel shuffle (view -> transpose -> contiguous -> view)
# Input: [B, C*2, H, W] where C = 20 or 40
# Output: [B, C*2, H, W] with channels shuffled
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4),
    ],
    key=['B', 'C', 'H', 'W'],
)
@triton.jit
def channel_shuffle_kernel(in_ptr, out_ptr,
                           B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
                           num_groups: tl.constexpr, group_size: tl.constexpr,
                           BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    batch_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    w_idx = tl.program_id(2)
    
    offs_h = h_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_w = w_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    mask_h = offs_h < H
    mask_w = offs_w < W
    mask = mask_h[:, None] & mask_w[None, :]
    
    # For each output channel c_out in range(C*2):
    # c_out from 0 to 2*C-1
    # Group size = C (40 or 80 total channels split into 2 groups of C/2)
    # channel_shuffle formula: new_c = (c_out // num_groups) * (C*2 // num_groups) + (c_out % num_groups)
    # For 80 channels with 2 groups: new_c = (c_out // 2) * 40 + (c_out % 2)
    
    for c_out in range(0, C * num_groups):
        group = c_out // group_size
        channel_in_group = c_out % group_size
        
        # Source channel in input
        src_c = channel_in_group * num_groups + group
        
        src_offset = batch_idx * (C * num_groups) * H * W + src_c * H * W + offs_h[:, None] * W + offs_w[None, :]
        dst_offset = batch_idx * (C * num_groups) * H * W + c_out * H * W + offs_h[:, None] * W + offs_w[None, :]
        
        val = tl.load(in_ptr + src_offset, mask=mask, other=0.0)
        tl.store(out_ptr + dst_offset, val, mask=mask)


# Kernel 6: Chunk (split along channel dimension)
@triton.jit
def chunk_kernel(in_ptr, out1_ptr, out2_ptr,
                 B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
                 BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    batch_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    w_idx = tl.program_id(2)
    
    offs_h = h_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_w = w_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    mask_h = offs_h < H
    mask_w = offs_w < W
    mask = mask_h[:, None] & mask_w[None, :]
    
    for c in range(C):
        in_offset = batch_idx * (C * 2) * H * W + c * H * W + offs_h[:, None] * W + offs_w[None, :]
        out1_offset = batch_idx * C * H * W + c * H * W + offs_h[:, None] * W + offs_w[None, :]
        
        val = tl.load(in_ptr + in_offset, mask=mask, other=0.0)
        tl.store(out1_ptr + out1_offset, val, mask=mask)
        
        in_offset2 = batch_idx * (C * 2) * H * W + (C + c) * H * W + offs_h[:, None] * W + offs_w[None, :]
        out2_offset = batch_idx * C * H * W + c * H * W + offs_h[:, None] * W + offs_w[None, :]
        
        val2 = tl.load(in_ptr + in_offset2, mask=mask, other=0.0)
        tl.store(out2_ptr + out2_offset, val2, mask=mask)


@torch.fx.wrap
def triton_forward(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    # Get shapes
    B = in_2.shape[0]
    
    # Step 1: Conv + Sigmoid + Mul for tmp_4
    # Shapes:
    # in_0: [40] bias
    # in_1: [40, 10, 1, 1] weight
    # in_6: [B, 10, 1, 1] input
    # in_5: [B, 40, 32, 24] multiply input
    
    # Conv1x1: in_6[B,10,1,1] * in_1[40,10,1,1] + in_0[40] -> conv_out[B,40,1,1]
    conv_out = torch.empty((B, 40, 1, 1), dtype=in_2.dtype, device=in_2.device)
    BLOCK_SIZE = 64
    num_programs = (B * 40 + BLOCK_SIZE - 1) // BLOCK_SIZE
    conv1x1_bias_kernel[(num_programs,)](
        in_6, in_1, in_0, conv_out,
        B, 10, 40, BLOCK_SIZE
    )
    
    # Sigmoid: conv_out[B,40,1,1] -> sigmoid_out[B,40,1,1]
    sigmoid_out = torch.empty_like(conv_out)
    num_elements = B * 40
    num_programs_sigmoid = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    sigmoid_kernel[(num_programs_sigmoid,)](
        conv_out, sigmoid_out, num_elements, BLOCK_SIZE
    )
    
    # Broadcast multiply: sigmoid_out[B,40,1,1] * in_5[B,40,32,24] -> tmp_4[B,40,32,24]
    tmp_4 = torch.empty((B, 40, 32, 24), dtype=in_2.dtype, device=in_2.device)
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    H_grid = (32 + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    W_grid = (24 + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    broadcast_mul_kernel[(B, H_grid, W_grid)](
        in_5, sigmoid_out, tmp_4,
        B, 40, 32, 24, BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    # Step 2: First channel shuffle path
    # tmp_5 = cat(in_2, in_4, dim=1) -> [B, 40, 64, 48]
    tmp_5 = torch.empty((B, 40, 64, 48), dtype=in_2.dtype, device=in_2.device)
    H_grid = (64 + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    W_grid = (48 + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    concat_channel_kernel[(B, H_grid, W_grid)](
        in_2, in_4, tmp_5,
        B, 20, 20, 64, 48, BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    # Channel shuffle: [B, 40, 64, 48] -> [B, 40, 64, 48]
    # (view [B, 2, 20, 64, 48] -> transpose [B, 20, 2, 64, 48] -> contiguous -> view [B, 40, 64, 48])
    tmp_10 = torch.empty((B, 40, 64, 48), dtype=in_2.dtype, device=in_2.device)
    H_grid = (64 + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    W_grid = (48 + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    channel_shuffle_kernel[(B, H_grid, W_grid)](
        tmp_5, tmp_10,
        B, 20, 64, 48, 2, 20, BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    # Chunk tmp_10: [B, 40, 64, 48] -> tmp_16[B, 20, 64, 48], tmp_17[B, 20, 64, 48]
    tmp_16 = torch.empty((B, 20, 64, 48), dtype=in_2.dtype, device=in_2.device)
    tmp_17 = torch.empty((B, 20, 64, 48), dtype=in_2.dtype, device=in_2.device)
    H_grid = (64 + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    W_grid = (48 + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    chunk_kernel[(B, H_grid, W_grid)](
        tmp_10, tmp_16, tmp_17,
        B, 20, 64, 48, BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    # Step 3: Second channel shuffle path
    # tmp_6 = cat(in_3, tmp_4, dim=1) -> [B, 80, 32, 24]
    tmp_6 = torch.empty((B, 80, 32, 24), dtype=in_2.dtype, device=in_2.device)
    H_grid = (32 + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    W_grid = (24 + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    concat_channel_kernel[(B, H_grid, W_grid)](
        in_3, tmp_4, tmp_6,
        B, 40, 40, 32, 24, BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    # Channel shuffle: [B, 80, 32, 24] -> [B, 80, 32, 24]
    # (view [B, 2, 40, 32, 24] -> transpose [B, 40, 2, 32, 24] -> contiguous -> view [B, 80, 32, 24])
    tmp_14 = torch.empty((B, 80, 32, 24), dtype=in_2.dtype, device=in_2.device)
    H_grid = (32 + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    W_grid = (24 + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    channel_shuffle_kernel[(B, H_grid, W_grid)](
        tmp_6, tmp_14,
        B, 40, 32, 24, 2, 40, BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    # Chunk tmp_14: [B, 80, 32, 24] -> tmp_19[B, 40, 32, 24], tmp_20[B, 40, 32, 24]
    tmp_19 = torch.empty((B, 40, 32, 24), dtype=in_2.dtype, device=in_2.device)
    tmp_20 = torch.empty((B, 40, 32, 24), dtype=in_2.dtype, device=in_2.device)
    H_grid = (32 + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    W_grid = (24 + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    chunk_kernel[(B, H_grid, W_grid)](
        tmp_14, tmp_19, tmp_20,
        B, 40, 32, 24, BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    return (tmp_16, tmp_19, tmp_17, tmp_20)


def replacement_func():
    return triton_forward