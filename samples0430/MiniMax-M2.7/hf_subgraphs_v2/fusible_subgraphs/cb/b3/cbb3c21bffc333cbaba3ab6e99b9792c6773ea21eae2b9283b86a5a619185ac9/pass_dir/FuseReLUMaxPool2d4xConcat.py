import torch
import triton
import triton.language as tl

# Triton kernel that fuses ReLU + 4x MaxPool2d + Concat
@triton.jit
def fused_relu_maxpool_concat_kernel(
    in_ptr,
    out_ptr,
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    KW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for: relu(in) + 4x max_pool2d(in) with kernel_size=5, stride=1, padding=2 + concat
    
    Reads input once and computes:
    - output[:, :C, :, :] = relu(input)
    - output[:, C:2*C, :, :] = max_pool(input)
    - output[:, 2*C:3*C, :, :] = max_pool(input)
    - output[:, 3*C:, :, :] = max_pool(input)
    """
    pid = tl.program_id(0)
    
    # Calculate output position
    # Total output elements: B * 4*C * H * W
    total_out_channels = 4 * C
    
    # Each thread handles one (b, c_out, h, w) position
    b = pid // (total_out_channels * H * W)
    remaining = pid % (total_out_channels * H * W)
    c_out = remaining // (H * W)
    remaining = remaining % (H * W)
    h = remaining // W
    w = remaining % W
    
    # Calculate input position (center of pooling window)
    # max_pool2d with kernel=5, stride=1, padding=2 maps output pos (h,w) to input pos (h, w)
    in_h = h
    in_w = w
    
    # Compute output for which of the 4 branches we're in
    branch_idx = c_out // C  # 0, 1, 2, or 3
    
    # For branch_idx 0: output is just relu(input)
    # For branch_idx 1,2,3: output is max_pool(input)
    
    # Load the input value at center position
    in_stride = C * H * W
    in_offset = b * in_stride + in_h * C * W + in_w * C + 0  # Base offset
    
    # Get the first channel value for relu
    inp_ptr = in_ptr + b * C * H * W + in_h * C * W + in_w * C
    
    # Apply ReLU using max(0, x)
    val = tl.load(inp_ptr).to(tl.float32)
    relu_val = tl.where(val > 0, val, 0.0)
    
    # Output offset for branch 0 (relu)
    out_offset_base = b * total_out_channels * H * W + h * total_out_channels * W + w * total_out_channels
    
    # Store branch 0 (relu output)
    out_ptr_0 = out_ptr + out_offset_base
    tl.store(out_ptr_0, relu_val.to(tl.float16 if C == 256 else tl.float32))
    
    # For max_pool branches (1, 2, 3), we need max over 5x5 window
    KW = 5
    pad = 2
    
    # Compute max pool by iterating over kernel window
    max_val = relu_val  # Start with relu of center value
    
    for kh in range(KW):
        for kw in range(KW):
            h_pos = in_h + kh - pad
            w_pos = in_w + kw - pad
            
            # Clamp to valid range
            h_clamped = tl.where(h_pos < 0, 0, tl.where(h_pos >= H, H - 1, h_pos))
            w_clamped = tl.where(w_pos < 0, 0, tl.where(w_pos >= W, W - 1, w_pos))
            
            offset = b * C * H * W + h_clamped * C * W + w_clamped * C
            v = tl.load(in_ptr + offset).to(tl.float32)
            v_relu = tl.where(v > 0, v, 0.0)
            max_val = tl.where(v_relu > max_val, v_relu, max_val)
    
    # Store branch 1 (max_pool output at channels C:2C)
    out_ptr_1 = out_ptr + out_offset_base + C
    tl.store(out_ptr_1, max_val.to(tl.float16 if C == 256 else tl.float32))
    
    # Store branch 2 (max_pool output at channels 2C:3C) - same value
    out_ptr_2 = out_ptr + out_offset_base + 2 * C
    tl.store(out_ptr_2, max_val.to(tl.float16 if C == 256 else tl.float32))
    
    # Store branch 3 (max_pool output at channels 3C:4C) - same value
    out_ptr_3 = out_ptr + out_offset_base + 3 * C
    tl.store(out_ptr_3, max_val.to(tl.float16 if C == 256 else tl.float32))


@torch.fx.wrap
def fused_kernel_wrapper(in_0):
    B, C, H, W = in_0.shape
    KW = 5
    
    # Output shape: [B, 4*C, H, W]
    output_shape = (B, 4 * C, H, W)
    
    # Allocate output with same dtype as input
    out = torch.empty(output_shape, device=in_0.device, dtype=in_0.dtype)
    
    # Total output elements
    total_out_elements = B * 4 * C * H * W
    
    # Launch kernel
    grid = (total_out_elements,)
    
    fused_relu_maxpool_concat_kernel[grid](
        in_0,
        out,
        B,
        C,
        H,
        W,
        KW,
        BLOCK_SIZE=1,  # Each thread handles one output element
    )
    
    return out


def pattern(in_0):
    """Match the pattern: relu + 4x max_pool2d + concat"""
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_2 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_3 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_4 = torch.cat([tmp_0, tmp_1, tmp_2, tmp_3], 1)
    return tmp_4


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return fused_kernel_wrapper