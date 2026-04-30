import torch
import triton
import triton.language as tl


@triton.jit
def fused_sigmoid_mul_contiguous_kernel(
    sigmoid_out_ptr,
    in_2_ptr,
    out_ptr,
    sigmoid_stride,
    in_2_stride_n,
    in_2_stride_c,
    in_2_stride_h,
    in_2_stride_w,
    N,
    C,
    H,
    W,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that computes:
    1. sigmoid (already computed, stored in sigmoid_out)
    2. view (implicit via broadcasting in this kernel)
    3. multiply sigmoid * in_2
    4. store result contiguously
    
    Uses 1D grid for maximum parallelism.
    Each program handles one (n, c) pair and loops over all spatial positions.
    """
    # Get position in output tensor
    nc_idx = tl.program_id(0)
    
    # Decode n and c from nc_idx
    n = nc_idx // C
    c = nc_idx % C

    # Load sigmoid value once (same for all spatial positions of this channel)
    sigmoid_idx = c * sigmoid_stride
    sigmoid_val = tl.load(sigmoid_out_ptr + sigmoid_idx)
    
    # Compute base offset for in_2 tensor
    base_offset = n * in_2_stride_n + c * in_2_stride_c
    
    # Compute output base offset (contiguous [N, C, H, W])
    out_base = (n * C + c) * H * W
    
    # Loop over spatial positions - vectorized
    for hw_start in range(0, HW, BLOCK_SIZE):
        hw_block = tl.arange(0, BLOCK_SIZE)
        offsets = hw_start + hw_block
        mask = offsets < HW
        
        # Compute h, w for this offset
        h_offsets = offsets // W
        w_offsets = offsets % W
        
        # Compute flat offsets for in_2
        in_2_offsets = base_offset + h_offsets * in_2_stride_h + w_offsets * in_2_stride_w
        
        # Compute flat offsets for output
        out_offsets = out_base + h_offsets * W + w_offsets
        
        # Load, multiply, and store
        in_2_vals = tl.load(in_2_ptr + in_2_offsets, mask=mask, other=0.0)
        result = sigmoid_val * in_2_vals
        tl.store(out_ptr + out_offsets, result, mask=mask)


@torch.fx.wrap
def fused_sigmoid_mul_contiguous_wrapper(sigmoid_output, in_2):
    """
    Wrapper function to launch the fused sigmoid-mul-contiguous kernel.
    
    Args:
        sigmoid_output: Tensor of shape [1, C, 1, 1] - sigmoid activation from conv
        in_2: Tensor of shape [N, C, H, W] - to be multiplied with sigmoid
    
    Returns:
        Tensor of shape [N, C, H, W], contiguous memory
    """
    N = in_2.shape[0]
    C = in_2.shape[1]
    H = in_2.shape[2]
    W = in_2.shape[3]
    HW = H * W
    
    # Allocate output tensor in [N, C, H, W] format (contiguous)
    out = torch.empty((N, C, H, W), dtype=in_2.dtype, device=in_2.device)
    
    # Get strides for in_2
    stride_n = in_2.stride(0)
    stride_c = in_2.stride(1)
    stride_h = in_2.stride(2)
    stride_w = in_2.stride(3)
    
    # Get strides for sigmoid_output (shape [1, C, 1, 1])
    sigmoid_stride_c = sigmoid_output.stride(1)  # This should be 1
    
    # Block size for processing spatial positions
    BLOCK_SIZE = 4096
    
    # Launch kernel with 1D grid
    # Grid: N*C - each program handles one (n, c) pair and loops over all HW
    grid = (N * C,)
    
    fused_sigmoid_mul_contiguous_kernel[grid](
        sigmoid_output,
        in_2,
        out,
        sigmoid_stride_c,
        stride_n,
        stride_c,
        stride_h,
        stride_w,
        N,
        C,
        H,
        W,
        HW,
        BLOCK_SIZE,
    )
    
    return out


def pattern(in_0, in_1, in_2, in_3):
    """
    Match the pattern:
    conv2d -> sigmoid -> view(1, -1, 1, 1) -> mul -> contiguous
    """
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 4)
    tmp_3 = torch.sigmoid(conv2d)
    tmp_4 = tmp_3.view(1, -1, 1, 1)
    tmp_5 = in_2 * tmp_4
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3):
    """
    Extract arguments needed for the replacement.
    
    For this fused operation, we need:
    - sigmoid_output (result of sigmoid on conv)
    - in_2 (tensor to multiply with)
    """
    # First compute the conv and sigmoid
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 4)
    sigmoid_output = torch.sigmoid(conv2d)
    
    return (sigmoid_output, in_2)


def replacement_func():
    """
    Return the replacement function that fuses sigmoid, view, mul, and contiguous.
    """
    return fused_sigmoid_mul_contiguous_wrapper