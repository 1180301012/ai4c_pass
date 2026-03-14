import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 16}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16}, num_stages=3, num_warps=8),
    ],
    key=['K'],
)
@triton.jit
def fused_conv_bn_kernel(
    # Conv inputs
    input_ptr, weight_ptr,
    # BN inputs
    running_mean_ptr, running_var_ptr, weight_bn_ptr, bias_bn_ptr,
    # Output
    output_ptr,
    # Shapes
    N, C, H, W,  # input shape
    K,  # output channels
    # BN params
    eps: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused Conv2d + BatchNorm kernel for inference.
    Conv: stride=1, padding=1, dilation=1, groups=1 (3x3 conv with padding)
    BN: training=False (use running mean/var)
    
    Grid: (K, N, H*W/block_n)
    Each program processes:
    - All input channels for one output channel (K dimension)
    - A tile of N*H*W positions
    """
    # Program handles one output channel
    pid_k = tl.program_id(0)
    pid_other = tl.program_id(1)
    
    if pid_k >= K:
        return
    
    # Calculate output position from pid_other
    # Tile the N*H*W dimension
    off_n = (pid_other * BLOCK_SIZE_N) // (H * W)
    off_hw = (pid_other * BLOCK_SIZE_N) % (H * W)
    off_h = off_hw // W
    off_w = off_hw % W
    
    # Load BN parameters for this output channel
    running_mean = tl.load(running_mean_ptr + pid_k)
    running_var = tl.load(running_var_ptr + pid_k)
    weight_bn = tl.load(weight_bn_ptr + pid_k)
    bias_bn = tl.load(bias_bn_ptr + pid_k)
    
    # Compute normalized weight: weight / sqrt(var + eps)
    inv_std = 1.0 / tl.sqrt(running_var + eps)
    normalized_weight = weight_bn * inv_std
    normalized_bias = bias_bn - running_mean * weight_bn * inv_std
    
    # Initialize output accumulator
    # Process BLOCK_SIZE_N positions in parallel
    offs_n = off_n + tl.arange(0, BLOCK_SIZE_M)[:, None]
    offs_h = off_h + tl.arange(0, BLOCK_SIZE_M)[:, None] * 0  # Same h for all in M dimension
    offs_w = off_w + tl.arange(0, BLOCK_SIZE_N)[None, :]
    
    # Mask for valid positions
    mask = (offs_n < N) & (offs_h < H) & (offs_w < W)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over input channels (K dimension reduction)
    for c in range(0, C, BLOCK_SIZE_K):
        # Load input tile: [BLOCK_SIZE_M, BLOCK_SIZE_K] for each of 3x3 kernel positions
        # We need to handle padding at boundaries
        
        # For each k in the tile
        c_offs = c + tl.arange(0, BLOCK_SIZE_K)
        c_mask = c_offs < C
        
        # Load weight tile: [BLOCK_SIZE_K, 9] 
        # Weight shape: [K, C, 3, 3] -> flatten to [K, C*9]
        weight_offs_base = pid_k * (C * 9) + c_offs * 9
        
        # For each input channel in the tile
        for kj in range(9):
            weight_offset = weight_offs_base + kj
            w_vals = tl.load(weight_ptr + weight_offset, mask=c_mask, other=0.0)
            w_vals = w_vals * normalized_weight  # Apply BN scale
            
            # For each position in output tile, load input and compute
            # Input has padding=1, so we need to handle boundary conditions
            # For 3x3 conv with stride=1, padding=1:
            # output[n, h, w] = sum over c, kh, kw of input[n, c, h+kh-1, w+kw-1] * weight[c, kh, kw]
            
            # Compute input offsets for each kernel position
            kh = kj // 3  # 0, 1, or 2
            kw = kj % 3   # 0, 1, or 2
            
            # Input position with padding
            # h_input = h + kh - 1 (since padding=1 means we add 1 to indices)
            h_input = offs_h - 1 + kh
            w_input = offs_w - 1 + kw
            
            # Mask for valid positions (within input bounds, accounting for padding=1)
            # Original input is [N, C, H, W], after padding becomes [N, C, H+2, W+2]
            # Valid padded indices are 1 to H (inclusive) for original H
            in_mask = (h_input >= 0) & (h_input < H) & (w_input >= 0) & (w_input < W)
            full_mask = in_mask & c_mask[None, :] & mask
            
            # Load input values
            # Flatten: n*C*H*W + c*H*W + h*W + w
            input_offsets = offs_n * C * H * W + c_offs[None, :] * H * W + h_input * W + w_input
            input_vals = tl.load(input_ptr + input_offsets, mask=full_mask, other=0.0)
            
            # Accumulate
            acc += input_vals * w_vals[None, :]
    
    # Add bias and store
    acc = acc + normalized_bias
    
    # Store output: [N, K, H, W] - flattened
    output_offsets = offs_n * K * H * W + pid_k * H * W + offs_h * W + offs_w
    tl.store(output_ptr + output_offsets, acc, mask=mask)


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """
    Match Conv2d + BatchNorm pattern.
    Conv: torch.conv2d(input, weight, bias=None, stride, padding, dilation, groups)
    BN: torch.nn.functional.batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_2
    tmp_3 = in_3
    tmp_4 = in_4
    tmp_5 = torch.conv2d(in_6, tmp_4, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_4 = None
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
    tmp_5 = tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
    tmp_7 = torch.nn.functional.avg_pool2d(in_5, 2, 2, 0, True, False, None)
    return (tmp_7, tmp_6)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6)


@torch.fx.wrap
def fused_conv_bn_kernel_wrapper(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """
    Wrapper function for the fused Conv2d + BatchNorm kernel.
    
    Inputs:
    - in_0: running_mean [K]
    - in_1: running_var [K] 
    - in_2: bias [K]
    - in_3: weight [K]
    - in_4: conv_weight [K, C, 3, 3]
    - in_5: avg_pool_input [N, C_in, H, W] - for avg_pool
    - in_6: conv_input [N, C, H, W] - for conv
    
    Note: Conv with padding=1 produces same size output
    """
    # Extract shapes
    N = in_6.shape[0]
    C = in_6.shape[1]
    H = in_6.shape[2]
    W = in_6.shape[3]
    
    K = in_4.shape[0]  # output channels
    
    eps = 1e-05
    
    # Output for conv+bn
    output_conv = torch.empty((N, K, H, W), dtype=torch.float32, device=in_6.device)
    
    # Grid: (K, ceil(N*H*W / BLOCK_SIZE_N))
    # Each program handles one output channel and a tile of N*H*W positions
    total_positions = N * H * W
    BLOCK_SIZE_N = 64  # Default, will be tuned by autotune
    num_other_programs = (total_positions + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    grid = (K, num_other_programs)
    
    fused_conv_bn_kernel[grid](
        in_6, in_4,
        in_0, in_1, in_3, in_2,
        output_conv,
        N, C, H, W,
        K,
        eps,
    )
    
    # Also compute avg_pool2d on in_5
    tmp_7 = torch.nn.functional.avg_pool2d(in_5, 2, 2, 0, True, False, None)
    
    return (tmp_7, output_conv)


def replacement_func():
    return fused_conv_bn_kernel_wrapper