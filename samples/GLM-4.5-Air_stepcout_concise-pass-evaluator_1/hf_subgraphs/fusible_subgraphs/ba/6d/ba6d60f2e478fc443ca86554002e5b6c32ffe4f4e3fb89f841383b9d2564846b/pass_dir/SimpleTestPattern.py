import torch
import triton
import triton.language as tl

def pattern(in_7, in_5, in_4, in_6, in_0, in_1, in_3, in_2):
    # Try to match the exact computation from one specific model
    tmp_6 = torch.conv2d(in_7, in_5, in_4, (1, 1), (0, 0), (1, 1), 1)
    tmp_7 = tmp_6.sigmoid()
    tmp_8 = in_6 * tmp_7
    tmp_9 = torch.nn.functional.relu(tmp_8, inplace=True)
    tmp_10 = torch.nn.functional.batch_norm(tmp_9, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    # Return only what the original function returns
    return tmp_9, tmp_10

def replacement_args(in_7, in_5, in_4, in_6, in_0, in_1, in_3, in_2):
    return (in_7, in_5, in_4, in_6, in_0, in_1, in_3, in_2)

@triton.jit
def simple_test_kernel(
    in_7_ptr, in_5_ptr, in_4_ptr, in_6_ptr,
    in_0_ptr, in_1_ptr, in_3_ptr, in_2_ptr,
    relu_out_ptr, bn_out_ptr,
    n_batch, n_in_channels, n_out_channels, sp_height, sp_width,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    # SE processing (small spatial dimensions)
    in_7_base = pid_b * n_in_channels * 1 * 1
    in_5_base = pid_c * n_in_channels
    
    # Load SE input and conv weights
    in_7_val = tl.load(in_7_ptr + in_7_base + pid_c)
    in_5_val = tl.load(in_5_ptr + in_5_base + pid_c)
    in_4_val = tl.load(in_4_ptr + pid_c)
    
    # Conv2D (SE reduction)
    se_conv = in_7_val * in_5_val + in_4_val
    se_sigmoid = 1.0 / (1.0 + tl.exp(-se_conv))
    
    # Load feature map and apply SE attention
    feature_base = pid_b * n_out_channels * sp_height * sp_width
    feature_ptr_local = in_6_ptr + feature_base + pid_c * sp_height * sp_width
    
    relu_base = pid_b * n_out_channels * sp_height * sp_width + pid_c * sp_height * sp_width
    relu_ptr_local = relu_out_ptr + relu_base
    bn_ptr_local = bn_out_ptr + relu_base
    
    # Load BN parameters
    bn_mean = tl.load(in_0_ptr + pid_c)
    bn_var = tl.load(in_1_ptr + pid_c)
    bn_weight = tl.load(in_3_ptr + pid_c)
    bn_bias = tl.load(in_2_ptr + pid_c)
    
    # Process spatial positions
    for h in range(sp_height):
        for w in range(sp_width):
            feat_addr = feature_ptr_local + h * sp_width + w
            relu_addr = relu_ptr_local + h * sp_width + w
            bn_addr = bn_ptr_local + h * sp_width + w
            
            # Load feature, apply SE attention, BN, and ReLU
            feat_val = tl.load(feat_addr)
            se_activated = feat_val * se_sigmoid
            bn_norm = (se_activated - bn_mean) / tl.sqrt(bn_var + eps)
            bn_out = bn_norm * bn_weight + bn_bias
            relu_out = tl.max(bn_out, 0.0)
            
            # Store results
            tl.store(bn_addr, bn_out)
            tl.store(relu_addr, relu_out)

@torch.fx.wrap
def simple_fusion_kernel(in_7, in_5, in_4, in_6, in_0, in_1, in_3, in_2):
    # Get input shapes
    n_batch, n_in_channels, _, _ = in_7.shape
    n_out_channels = in_5.shape[0]
    sp_channels, sp_height, sp_width = in_6.shape[1], in_6.shape[2], in_6.shape[3]
    
    # Create output tensors
    relu_out = torch.empty_like(in_6)
    bn_out = torch.empty_like(in_6)
    
    # Launch kernel
    BLOCK_SIZE = 64
    grid_b = n_batch
    grid_c = (n_out_channels + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    simple_test_kernel[grid_b, grid_c](
        in_7_ptr=in_7,
        in_5_ptr=in_5,
        in_4_ptr=in_4,
        in_6_ptr=in_6,
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        in_3_ptr=in_3,
        in_2_ptr=in_2,
        relu_out_ptr=relu_out,
        bn_out_ptr=bn_out,
        n_batch=n_batch,
        n_in_channels=n_in_channels,
        n_out_channels=n_out_channels,
        sp_height=sp_height,
        sp_width=sp_width,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return relu_out, bn_out

def replacement_func():
    return simple_fusion_kernel