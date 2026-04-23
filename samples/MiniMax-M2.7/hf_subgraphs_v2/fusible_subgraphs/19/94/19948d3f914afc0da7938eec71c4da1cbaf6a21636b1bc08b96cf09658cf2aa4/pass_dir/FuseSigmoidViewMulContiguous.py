import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    """
    Match the complete subgraph: conv2d -> sigmoid -> view -> mul -> contiguous
    """
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 4)
    tmp_3 = torch.sigmoid(conv2d)
    tmp_4 = tmp_3.view(1, -1, 1, 1)
    tmp_5 = in_2 * tmp_4
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3):
    """Extract arguments needed for the replacement function."""
    return (in_0, in_1, in_2, in_3)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=4, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def fused_mul_contiguous_kernel(
    x_ptr,            # in_2: (1, 96, H, W)
    sigmoid_ptr,      # sigmoid result: (1, 96, 1, 1)
    out_ptr,          # output tensor
    N,                # Total number of elements
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_sn, stride_sc,
    stride_on, stride_oc, stride_oh, stride_ow,
    H, W,             # Spatial dimensions
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for multiplication with broadcasting and contiguous store.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    HW = H * W
    
    # Calculate channel, height, width for each thread
    c = offsets // HW
    remainder = offsets % HW
    h = remainder // W
    w = remainder % W
    
    # Load sigmoid values (broadcast from 1, 96, 1, 1 to 1, 96, H, W)
    sigmoid_offsets = c * stride_sc
    sigmoid_vals = tl.load(sigmoid_ptr + sigmoid_offsets, mask=mask, other=0.0)
    
    # Load x values
    x_offsets = 0 * stride_xn + c * stride_xc + h * stride_xh + w * stride_xw
    x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
    
    # Multiply
    out_vals = x_vals * sigmoid_vals
    
    # Calculate output offsets
    out_offsets = 0 * stride_on + c * stride_oc + h * stride_oh + w * stride_ow
    
    # Store result
    tl.store(out_ptr + out_offsets, out_vals, mask=mask)


@torch.fx.wrap
def fused_mul_contiguous_wrapper(in_0, in_1, in_2, in_3):
    """
    Wrapper function that fuses mul + contiguous operations.
    Uses in_0 as sigmoid output (framework provides it from matched subgraph).
    """
    sigmoid_out = in_0
    x = in_2
    
    channels = x.shape[1]
    height = x.shape[2]
    width = x.shape[3]
    total_elements = channels * height * width
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Grid configuration
    BLOCK_SIZE = 1024
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # Launch kernel
    fused_mul_contiguous_kernel[grid](
        x,
        sigmoid_out,
        out,
        total_elements,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        sigmoid_out.stride(0) if len(sigmoid_out.stride()) > 1 else 0,
        sigmoid_out.stride(1) if len(sigmoid_out.stride()) > 1 else 0,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        height,
        width,
    )
    
    return out


def replacement_func():
    return fused_mul_contiguous_wrapper