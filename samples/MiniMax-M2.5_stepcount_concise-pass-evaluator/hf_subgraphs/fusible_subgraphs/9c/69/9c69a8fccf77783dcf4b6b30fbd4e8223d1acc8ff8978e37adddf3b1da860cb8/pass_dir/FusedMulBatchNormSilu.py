import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Pattern: Multiply -> BatchNorm -> SiLU
    This matches the exact computation:
    tmp_4 = in_5 * in_4
    tmp_5 = batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = silu(tmp_5)
    """
    # Element-wise multiplication with broadcasting (in_4 has shape [*, C, 1, 1])
    tmp_4 = in_5 * in_4
    # BatchNorm: input, running_mean, running_var, weight, bias, training, momentum, eps
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    # SiLU activation (inplace=True doesn't matter for pattern matching)
    tmp_6 = torch.nn.functional.silu(tmp_5, inplace=True)
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Extract the arguments needed for the fused kernel.
    in_0: running_mean
    in_1: running_var  
    in_2: bias
    in_3: weight
    in_4: sigmoid tensor (to be multiplied)
    in_5: input tensor
    """
    return (in_0, in_1, in_2, in_3, in_4, in_5)


@triton.autotune(
    configs=[
        # Try different block sizes for better occupancy
        triton.Config({'BLOCK_SIZE': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def fused_mul_bn_silu_kernel(
    # Input pointers
    in_5_ptr, in_4_ptr, in_0_ptr, in_1_ptr, in_2_ptr, in_3_ptr,
    # Output pointer
    out_ptr,
    # Sizes
    N, C, H, W,
    # Strides
    stride_in_5_n, stride_in_5_c, stride_in_5_h, stride_in_5_w,
    stride_in_4_n, stride_in_4_c, stride_in_4_h, stride_in_4_w,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: Multiply -> BatchNorm -> SiLU
    
    BatchNorm formula (training=False):
    y = (x - mean) / sqrt(var + eps) * weight + bias
    
    SiLU formula:
    y = x * sigmoid(x) = x / (1 + exp(-x))
    """
    # Each program processes a contiguous block of elements
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load input tensor in_5 (shape: [B, C, H, W])
    # Need to compute offsets for 4D tensor
    # Compute n, c, h, w indices from linear offset
    n = offsets // (C * H * W)
    c_remain = offsets % (C * H * W)
    c = c_remain // (H * W)
    hw_remain = c_remain % (H * W)
    h = hw_remain // W
    w = hw_remain % W
    
    # Compute strides for in_5
    in_5_offset = n * stride_in_5_n + c * stride_in_5_c + h * stride_in_5_h + w * stride_in_5_w
    in_5 = tl.load(in_5_ptr + in_5_offset, mask=mask, other=0.0)
    
    # Load in_4 (shape: [B, C, 1, 1] or [1, C, 1, 1] - broadcast)
    # Compute offset for in_4
    in_4_n = n if stride_in_4_n > 0 else 0
    in_4_offset = in_4_n * stride_in_4_n + c * stride_in_4_c
    in_4 = tl.load(in_4_ptr + in_4_offset, mask=mask, other=0.0)
    
    # Multiply: tmp_4 = in_5 * in_4
    tmp_4 = in_5 * in_4
    
    # Load BatchNorm parameters (shape: [C])
    mean = tl.load(in_0_ptr + c, mask=mask, other=0.0)
    var = tl.load(in_1_ptr + c, mask=mask, other=0.0)
    weight = tl.load(in_3_ptr + c, mask=mask, other=0.0)
    bias = tl.load(in_2_ptr + c, mask=mask, other=0.0)
    
    # BatchNorm: (x - mean) / sqrt(var + eps) * weight + bias
    eps = 1e-05
    normalized = (tmp_4 - mean) * weight / tl.sqrt(var + eps) + bias
    
    # SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    # Use more numerically stable computation
    sigmoid_neg = 1.0 / (1.0 + tl.exp(-normalized))
    result = normalized * sigmoid_neg
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_mul_bn_silu_kernel_wrapper(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Wrapper function that launches the fused Triton kernel.
    
    Args:
        in_0: running_mean [C]
        in_1: running_var [C] 
        in_2: bias [C]
        in_3: weight [C]
        in_4: sigmoid tensor [B, C, 1, 1] or [1, C, 1, 1]
        in_5: input tensor [B, C, H, W]
    
    Returns:
        Output tensor [B, C, H, W]
    """
    # Get input shape
    B, C, H, W = in_5.shape
    N = B * C * H * W  # Total number of elements
    
    # Allocate output
    out = torch.empty_like(in_5)
    
    # Choose block size based on total elements
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_mul_bn_silu_kernel[(num_programs,)](
        in_5_ptr=in_5,
        in_4_ptr=in_4,
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        in_2_ptr=in_2,
        in_3_ptr=in_3,
        out_ptr=out,
        N=N,
        C=C,
        H=H,
        W=W,
        stride_in_5_n=in_5.stride(0),
        stride_in_5_c=in_5.stride(1),
        stride_in_5_h=in_5.stride(2),
        stride_in_5_w=in_5.stride(3),
        stride_in_4_n=in_4.stride(0),
        stride_in_4_c=in_4.stride(1),
        stride_in_4_h=in_4.stride(2),
        stride_in_4_w=in_4.stride(3),
    )
    
    return out


def replacement_func():
    """Return the replacement function."""
    return fused_mul_bn_silu_kernel_wrapper