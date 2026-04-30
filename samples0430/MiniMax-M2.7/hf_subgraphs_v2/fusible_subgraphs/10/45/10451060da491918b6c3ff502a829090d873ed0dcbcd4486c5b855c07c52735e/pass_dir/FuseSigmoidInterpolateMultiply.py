import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Match the pattern: conv2d -> sigmoid
    
    Args:
        in_0: weight tensor [128, 960, 1, 1]
        in_1: input tensor [1, 960, 1, 4]
    """
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.sigmoid(conv2d)
    return tmp_2


def replacement_args(in_0, in_1):
    """
    Extract arguments needed for the fused replacement.
    """
    return (in_0, in_1)


@triton.jit
def fused_conv_sigmoid_kernel(
    weight_ptr, input_ptr, out_ptr,
    OC, IC, H, W,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: Conv2d -> Sigmoid
    
    Each program handles BLOCK_SIZE output elements. Uses loop to accumulate over IC.
    """
    pid = tl.program_id(0)
    
    # Block start index
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Each thread handles one output element
    oc = offsets // (H * W)
    oh = (offsets % (H * W)) // W
    ow = offsets % W
    
    # Initialize accumulator for each element
    conv_val = tl.zeros([BLOCK_SIZE], tl.float32)
    
    # Reduction loop over IC
    # Each iteration loads a vector of values (one per thread)
    for ic in range(IC):
        w_idx = oc * IC + ic  # This is a vector of indices
        i_idx = ic * H * W + oh * W + ow
        
        w_val = tl.load(weight_ptr + w_idx, mask=mask, other=0.0)
        i_val = tl.load(input_ptr + i_idx, mask=mask, other=0.0)
        conv_val = conv_val + w_val * i_val
    
    # Apply sigmoid
    sig_val = 1.0 / (1.0 + tl.exp(-conv_val))
    
    # Store result
    out_idx = oc * H * W + oh * W + ow
    tl.store(out_ptr + out_idx, sig_val, mask=mask)


@torch.fx.wrap
def fused_conv_sigmoid(weight, input_tensor):
    """
    Fused conv2d -> sigmoid kernel.
    """
    OC, IC, KH, KW = weight.shape
    B, _, H, W = input_tensor.shape
    
    out = torch.empty((B, OC, H, W), dtype=input_tensor.dtype, device=input_tensor.device)
    
    n_elements = OC * H * W
    BLOCK_SIZE = 8  # Small block for memory efficiency
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_conv_sigmoid_kernel[(num_programs,)](
        weight, input_tensor, out,
        OC, IC, H, W,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    """
    Return the fused kernel function.
    """
    return fused_conv_sigmoid