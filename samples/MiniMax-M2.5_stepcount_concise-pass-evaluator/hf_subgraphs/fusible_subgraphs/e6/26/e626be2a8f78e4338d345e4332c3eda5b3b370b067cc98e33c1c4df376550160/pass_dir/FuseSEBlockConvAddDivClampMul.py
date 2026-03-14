import torch
import triton
import triton.language as tl


# Stage 1: Compute excitation for each (b, c) pair
@triton.jit
def compute_excitation_kernel(
    in_3_ptr,  # [B, C_in, 1, 1]
    weight_ptr,  # [C_out, C_in, 1, 1]
    bias_ptr,  # [C_out]
    excitation_ptr,  # [B, C_out]
    B: tl.constexpr,
    C_in: tl.constexpr,
    C_out: tl.constexpr,
):
    """
    Compute excitation: conv(in_3, weight, bias) -> [B, C_out]
    Grid: B * C_out (one program per (b, c) pair)
    """
    # Grid: (B * C_out,)
    pid = tl.program_id(0)
    b = pid // C_out
    c = pid % C_out
    
    # Compute excitation[b, c] = sum_i(in_3[b,i] * weight[c,i]) + bias[c]
    excitation = 0.0
    for ci in range(C_in):
        in_3_idx = b * C_in + ci
        in_3_val = tl.load(in_3_ptr + in_3_idx).to(tl.float32)
        
        weight_idx = c * C_in + ci
        weight_val = tl.load(weight_ptr + weight_idx).to(tl.float32)
        
        excitation += in_3_val * weight_val
    
    # Add bias
    bias_val = tl.load(bias_ptr + c).to(tl.float32)
    excitation += bias_val
    
    # Store excitation
    exc_idx = b * C_out + c
    tl.store(excitation_ptr + exc_idx, excitation)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=4, num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def multiply_output_kernel(
    excitation_ptr,  # [B, C_out]
    in_2_ptr,  # [B, C_out, H, W]
    out_ptr,  # [B, C_out, H, W]
    B: tl.constexpr,
    C_out: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Multiply excitation [B, C_out] with in_2 [B, C_out, H, W] and output.
    Grid: (ceil(N / BLOCK_SIZE),)
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    for i in range(BLOCK_SIZE):
        offset = block_start + i
        valid = offset < N
        
        # Calculate (b, c, h, w) from offset
        # offset = ((b * C_out + c) * H + h) * W + w
        tmp = offset
        w = tmp % W
        tmp = tmp // W
        h = tmp % H
        tmp = tmp // H
        c = tmp % C_out
        b = tmp // C_out
        
        # Load excitation[b, c]
        exc_idx = b * C_out + c
        excitation = tl.load(excitation_ptr + exc_idx).to(tl.float32)
        
        # Apply activation: (x + 1.0) / 2.0, clamp to [0, 1]
        excitation = (excitation + 1.0) / 2.0
        excitation = tl.minimum(tl.maximum(excitation, 0.0), 1.0)
        
        # Load in_2[b, c, h, w]
        in_2_idx = b * C_out * H * W + c * H * W + h * W + w
        in_2_val = tl.load(in_2_ptr + in_2_idx).to(tl.float32)
        
        # Multiply
        result = in_2_val * excitation
        
        # Store
        tl.store(out_ptr + offset, result, mask=valid)


def pattern(in_0, in_1, in_2, in_3):
    """
    Match the SE block computation pattern:
    1. Conv2d with bias
    2. Add 1.0
    3. Divide by 2.0  
    4. Clamp to [0, 1]
    5. Multiply with feature map
    """
    # Conv2d: tmp_2 = conv(in_3, in_1, in_0)
    tmp_2 = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    # Add 1.0: tmp_3 = tmp_2 + 1.0
    tmp_3 = tmp_2 + 1.0
    # Divide by 2.0: tmp_4 = tmp_3 / 2.0
    tmp_4 = tmp_3 / 2.0
    # Clamp to [0, 1]: tmp_5 = clamp(tmp_4, 0, 1)
    tmp_5 = tmp_4.clamp_(0.0, 1.0)
    # Multiply: tmp_6 = in_2 * tmp_5
    tmp_6 = in_2 * tmp_5
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3):
    """Extract the arguments needed for the replacement kernel."""
    return (in_0, in_1, in_2, in_3)


@torch.fx.wrap
def se_block_fused_kernel_wrapper(in_0, in_1, in_2, in_3):
    """
    Wrapper function that launches the fused SE block kernel.
    
    Args:
        in_0: bias tensor [400]
        in_1: weight tensor [400, 100, 1, 1]
        in_2: feature map [B, 400, H, W]
        in_3: SE excitation input [B, 100, 1, 1]
    
    Returns:
        Output tensor [B, 400, H, W]
    """
    B = in_3.shape[0]
    C_in = in_3.shape[1]
    C_out = in_1.shape[0]
    H = in_2.shape[2]
    W = in_2.shape[3]
    
    # Allocate excitation buffer
    excitation = torch.empty((B, C_out), dtype=torch.float32, device=in_2.device)
    
    # Stage 1: Compute excitation for each (b, c) pair
    # Grid: (B * C_out,)
    compute_excitation_kernel[(B * C_out,)](
        in_3, in_1, in_0, excitation,
        B, C_in, C_out,
    )
    
    # Allocate output
    output = torch.empty((B, C_out, H, W), dtype=torch.float32, device=in_2.device)
    
    # Stage 2: Multiply excitation with in_2 and apply activation
    # Grid: (ceil(N / BLOCK_SIZE),)
    N = B * C_out * H * W
    grid = ((N + 1024 - 1) // 1024,)
    
    multiply_output_kernel[grid](
        excitation, in_2, output,
        B, C_out, H, W, N,
    )
    
    return output


def replacement_func():
    """Return the replacement function."""
    return se_block_fused_kernel_wrapper