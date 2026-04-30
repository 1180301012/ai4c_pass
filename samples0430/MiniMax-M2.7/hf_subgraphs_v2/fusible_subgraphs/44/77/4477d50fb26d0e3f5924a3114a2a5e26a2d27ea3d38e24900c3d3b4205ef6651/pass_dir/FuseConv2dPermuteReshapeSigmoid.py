import torch
import triton
import triton.language as tl

@triton.jit
def fused_conv_permute_reshape_sigmoid_kernel(
    # Conv2d inputs
    in_2_ptr, in_1_ptr, in_0_ptr,
    # Output pointer
    out_ptr,
    # Conv2d params
    stride_h: tl.constexpr, stride_w: tl.constexpr,
    padding_h: tl.constexpr, padding_w: tl.constexpr,
    dilation_h: tl.constexpr, dilation_w: tl.constexpr,
    groups: tl.constexpr,
    # Tensor shapes
    B: tl.constexpr, C_in: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    C_out: tl.constexpr, K_h: tl.constexpr, K_w: tl.constexpr,
    # Output reshape params
    out_batch: tl.constexpr, out_groups: tl.constexpr
):
    """
    Fused kernel that performs:
    1. Conv2d with 1x1 kernel (stride=1, padding=0)
    2. Permute (N, C, H, W) -> (N, H, W, C)
    3. Reshape to (batch, H*W, groups)
    4. Sigmoid activation
    """
    # Get global thread index
    pid = tl.program_id(0)
    
    # Calculate total output elements: out_batch * (H*W) * out_groups
    total_out_elements = out_batch * H * W * out_groups
    
    # Bounds check
    if pid >= total_out_elements:
        return
    
    # Calculate output indices
    # Flattened output: [batch_idx * (H*W*out_groups) + hw_idx * out_groups + group_idx]
    group_idx = pid % out_groups
    temp = pid // out_groups
    hw_idx = temp % (H * W)
    batch_idx = temp // (H * W)
    
    # Calculate H, W indices
    h_idx = hw_idx // W
    w_idx = hw_idx % W
    
    # The permute+reshape means:
    # output[batch_idx, hw_idx, group_idx] corresponds to:
    # conv_out[batch_idx, group_idx, h_idx, w_idx]
    
    # Conv2d computation:
    # output[b, c_out, h, w] = sum over c_in of input[b, c_in, h*stride, w*stride] * weight[c_out, c_in, 0, 0]
    # Then + bias[c_out]
    
    # Since dilation=1, stride=1, padding=0 for all cases:
    # conv_h = h_idx
    # conv_w = w_idx
    
    # For 1x1 conv, we iterate over input channels
    acc = tl.load(in_0_ptr + group_idx).to(tl.float32)
    
    # Unroll over input channels for better performance
    for c_in_idx in range(C_in):
        # Load input value: in_2[b, c_in, h, w]
        inp_offset = batch_idx * C_in * H * W + c_in_idx * H * W + h_idx * W + w_idx
        inp_val = tl.load(in_2_ptr + inp_offset).to(tl.float32)
        
        # Load weight: in_1[group_idx, c_in_idx, 0, 0] = in_1[group_idx * C_in + c_in_idx]
        weight_offset = group_idx * C_in + c_in_idx
        weight_val = tl.load(in_1_ptr + weight_offset).to(tl.float32)
        
        acc += inp_val * weight_val
    
    # Apply sigmoid: sigmoid(x) = 1 / (1 + exp(-x))
    acc_sigmoid = 1.0 / (1.0 + tl.exp(-acc))
    
    # Store output (convert back to original dtype)
    out_offset = batch_idx * H * W * out_groups + hw_idx * out_groups + group_idx
    tl.store(out_ptr + out_offset, acc_sigmoid.to(tl.bfloat16))


@torch.fx.wrap
def fused_conv_permute_reshape_sigmoid(in_0, in_1, in_2, out_shape):
    """
    Wrapper for the fused kernel.
    Performs conv2d + permute + reshape + sigmoid in a single kernel.
    """
    # Get shapes
    B, C_in, H, W = in_2.shape  # Input: [B, 512, 64, 128]
    C_out = out_shape[2]  # Number of groups (output channels from conv)
    
    # Allocate output
    out = torch.empty(out_shape, dtype=in_2.dtype, device=in_2.device)
    
    # Calculate total elements and grid size
    total_elements = out.numel()
    
    # Launch kernel
    fused_conv_permute_reshape_sigmoid_kernel[(total_elements,)](
        in_2, in_1, in_0,
        out,
        1, 1,  # stride
        0, 0,  # padding
        1, 1,  # dilation
        1,  # groups
        B, C_in, H, W,
        C_out, 1, 1,  # kernel size (unused but required)
        out_shape[0], out_shape[2]
    )
    
    return out


def pattern(in_0, in_1, in_2):
    """
    Match: conv2d -> permute -> reshape -> sigmoid
    """
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.permute(0, 2, 3, 1)
    tmp_4 = tmp_3.reshape(conv2d.shape[0], -1, in_1.shape[0])
    tmp_5 = torch.nn.functional.sigmoid(tmp_4)
    return tmp_5


def replacement_args(in_0, in_1, in_2):
    """
    Extract arguments needed for the fused kernel.
    """
    # Calculate output shape
    B = in_2.shape[0]
    G = in_1.shape[0]  # Number of output channels = number of groups
    H = in_2.shape[2]
    W = in_2.shape[3]
    out_shape = (B, H * W, G)
    
    return (in_0, in_1, in_2, out_shape)


def replacement_func():
    """
    Return the fused kernel function.
    """
    return fused_conv_permute_reshape_sigmoid