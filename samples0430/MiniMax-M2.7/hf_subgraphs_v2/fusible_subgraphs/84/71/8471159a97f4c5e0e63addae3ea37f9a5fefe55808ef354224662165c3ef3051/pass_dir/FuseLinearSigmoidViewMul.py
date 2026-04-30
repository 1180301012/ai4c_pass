import torch
import triton
import triton.language as tl

@triton.jit
def fused_linear_sigmoid_mul_kernel(
    x_ptr, weight_ptr, bias_ptr, feat_ptr,
    out_ptr,
    batch_stride_x, feat_in_stride, bias_stride,
    batch_stride_f, chan_stride_f, H_stride, W_stride,
    out_batch_stride, out_chan_stride, out_H_stride, out_W_stride,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: linear -> sigmoid -> view -> mul
    x: [batch, 8] - input features
    weight: [64, 8] - linear weights (transposed internally)
    bias: [64] - linear bias
    feat: [batch, 64, H, W] - feature tensor to scale
    output: [batch, 64, H, W]
    """
    # Get program id for batching
    pid = tl.program_id(0)
    
    # Calculate batch index and spatial indices
    batch_idx = pid // (H * W)
    spatial_idx = pid % (H * W)
    h_idx = spatial_idx // W
    w_idx = spatial_idx % W
    
    # Compute linear for all 64 output channels
    # Each program handles all 64 channels for one spatial position
    linear_results = tl.zeros((64,), dtype=tl.float32)
    
    for k in range(8):
        # Load x[batch, k]
        x_offset = batch_idx * batch_stride_x + k * feat_in_stride
        x_val = tl.load(x_ptr + x_offset)
        
        # Load weight[:, k] for all 64 channels
        for c in range(64):
            weight_offset = c * 64 + k  # weight[c, k]
            w_val = tl.load(weight_ptr + weight_offset)
            linear_results = linear_results + x_val * w_val
    
    # Add bias
    for c in range(64):
        bias_offset = c * bias_stride
        bias_val = tl.load(bias_ptr + bias_offset)
        linear_results = linear_results + bias_val
    
    # Apply sigmoid
    sigmoid_results = 1.0 / (1.0 + tl.exp(-linear_results))
    
    # Multiply with feature tensor and store
    # feat[batch, c, h, w] * sigmoid_results[c]
    for c in range(64):
        feat_offset = (batch_idx * batch_stride_f + 
                       c * chan_stride_f + 
                       h_idx * H_stride + 
                       w_idx * W_stride)
        feat_val = tl.load(feat_ptr + feat_offset)
        
        out_offset = (batch_idx * out_batch_stride + 
                      c * out_chan_stride + 
                      h_idx * out_H_stride + 
                      w_idx * out_W_stride)
        
        out_val = feat_val * sigmoid_results[c]
        tl.store(out_ptr + out_offset, out_val)


@torch.fx.wrap
def fused_linear_sigmoid_mul(x, weight, bias, feat):
    """
    Fused linear + sigmoid + view + element-wise multiply.
    x: [batch, 8]
    weight: [64, 8]
    bias: [64]
    feat: [batch, 64, H, W]
    returns: [batch, 64, H, W]
    """
    B, C, H, W = feat.shape
    batch, feat_in = x.shape
    
    # Allocate output
    out = torch.empty_like(feat)
    
    # Launch grid: one program per spatial position (B * H * W)
    grid = (B * H * W,)
    
    # Compute strides
    x_stride_b = x.stride(0)
    x_stride_f = x.stride(1)
    
    feat_stride_b = feat.stride(0)
    feat_stride_c = feat.stride(1)
    feat_stride_h = feat.stride(2)
    feat_stride_w = feat.stride(3)
    
    out_stride_b = out.stride(0)
    out_stride_c = out.stride(1)
    out_stride_h = out.stride(2)
    out_stride_w = out.stride(3)
    
    fused_linear_sigmoid_mul_kernel[grid](
        x, weight, bias, feat, out,
        x_stride_b, x_stride_f, 1,  # bias stride
        feat_stride_b, feat_stride_c, feat_stride_h, feat_stride_w,
        out_stride_b, out_stride_c, out_stride_h, out_stride_w,
        B, C, H, W,
        BLOCK_SIZE=1,
    )
    
    return out


def pattern(in_0, in_1, in_2, in_3):
    """
    Match the pattern: linear -> sigmoid -> view -> mul
    in_0: bias [64]
    in_1: weight [64, 8]
    in_2: input [batch, 8]
    in_3: feature [batch, 64, H, W]
    """
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.sigmoid(linear)
    tmp_4 = tmp_3.view(1, 64, 1, 1)
    tmp_5 = in_3 * tmp_4
    return tmp_5


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return fused_linear_sigmoid_mul