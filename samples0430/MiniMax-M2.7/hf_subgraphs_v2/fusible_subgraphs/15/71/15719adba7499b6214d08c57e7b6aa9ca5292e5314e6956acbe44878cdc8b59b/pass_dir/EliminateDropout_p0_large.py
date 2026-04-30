import torch
import triton
import triton.language as tl

# Pattern to match for swin_arocr_tiny: conv2d -> flatten(2) -> transpose(1,2) -> layer_norm -> dropout(p=0.0) -> view -> pad -> view -> permute
def pattern(in_0, in_1, in_2, in_3, in_4):
    # Conv2D with bias (groups=1), stride (4,4) for 1024x1024 input
    conv2d = torch.conv2d(in_0, in_4, in_3, (4, 4), (0, 0), (1, 1), 1)
    # Flatten spatial dimensions
    tmp_6 = conv2d.flatten(2)
    # Transpose to sequence format (B, N, C)
    tmp_7 = tmp_6.transpose(1, 2)
    # LayerNorm with 96 channels
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (96,), in_2, in_1, 1e-05)
    # Dropout with p=0.0 - this is a no-op!
    tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False)
    # View to spatial format [B, 256, 256, 96]
    tmp_10 = tmp_9.view(1, 256, 256, 96)
    # Pad (no-op with all zeros)
    tmp_11 = torch.nn.functional.pad(tmp_10, (0, 0, 0, 0, 0, 0), 'constant', None)
    # View to split spatial dimensions [B, 32, 8, 32, 8, 96]
    tmp_12 = tmp_11.view(1, 32, 8, 32, 8, 96)
    # Permute to final format [B, 32, 32, 8, 8, 96]
    tmp_13 = tmp_12.permute(0, 1, 3, 2, 4, 5)
    return (tmp_9, tmp_13)

def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


@torch.fx.wrap
def optimized_reshape_permute_large(in_0, in_1, in_2, in_3, in_4):
    """Optimized kernel for large input case (swin_arocr_tiny):
    1. Performs Conv2D + Flatten + Transpose + LayerNorm + Dropout elimination
    2. Optimizes view + pad + view + permute into a single efficient operation
    """
    # Conv2D with bias, stride (4,4)
    conv_out = torch.conv2d(in_0, in_4, in_3, (4, 4), (0, 0), (1, 1), 1)
    
    # Input: [B=1, C_in=3, H=1024, W=1024]
    # Conv: [B=1, C_out=96, H_out=256, W_out=256] (stride 4)
    B, C_out, H_out, W_out = conv_out.shape
    
    # LayerNorm parameters
    normalized_shape = (96,)
    weight = in_2  # layer_norm weight
    bias = in_1    # layer_norm bias
    eps = 1e-05
    
    # Manual LayerNorm on [B, H*W, C]
    # First reshape conv_out to [B, C, H*W], then transpose to [B, H*W, C]
    conv_reshaped = conv_out.reshape(B, C_out, H_out * W_out)  # [B, C, H*W]
    conv_transposed = conv_reshaped.permute(0, 2, 1)  # [B, H*W, C]
    
    # Compute LayerNorm
    ln_out = torch.nn.functional.layer_norm(conv_transposed, normalized_shape, weight, bias, eps)
    
    # Dropout with p=0.0 is a no-op
    dropout_out = ln_out
    
    # Now compute the permuted output efficiently
    # Original view sequence: [B, N, C] -> [B, H, W, C] -> [B, h1, s1, h2, s2, C] -> permute
    # Parameters: H=W=256, h1=h2=32, s1=s2=8
    h1, s1, h2, s2 = 32, 8, 32, 8
    H_new, W_new = h1 * s1, h2 * s2
    C_new = C_out
    
    # Reshape [B, H*W, C] -> [B, H, W, C]
    view1_out = conv_transposed.reshape(B, H_new, W_new, C_new)
    
    # Reshape to [B, h1, s1, h2, s2, C]
    view2_out = view1_out.reshape(B, h1, s1, h2, s2, C_new)
    
    # Permute [B, h1, s1, h2, s2, C] -> [B, h1, h2, s1, s2, C]
    perm_out = view2_out.permute(0, 1, 3, 2, 4, 5)
    
    return (dropout_out, perm_out)


def replacement_func():
    return optimized_reshape_permute_large