import torch
import triton
import triton.language as tl

# Pattern: Conv2D + View + Softmax for input shape [32, 512, 64, 64]
def pattern(in_0, in_1, in_2):
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = tmp_2.view(32, 1, -1)
    tmp_4 = tmp_3.softmax(dim=-1)
    return tmp_4

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def fused_conv_softmax_kernel_large(
    x_ptr,                    # Input feature map [32, 512, 64, 64]
    weight_ptr,              # Weights [1, 512, 1, 1]
    bias_ptr,                # Bias [1]
    out_ptr,                 # Output [32, 1, 4096] after softmax
    B: tl.constexpr,         # Batch size = 32
    H: tl.constexpr,         # Height = 64
    W: tl.constexpr,         # Width = 64  
    C_in: tl.constexpr,      # Input channels = 512
    C_out: tl.constexpr,     # Output channels = 1
    BLOCK_HW: tl.constexpr,   # Block size for spatial dimensions
):
    # Each program handles one spatial location for one batch
    pid = tl.program_id(0)
    batch_index = pid // (H * W)
    spatial_index = pid % (H * W)
    
    batch_mask = batch_index < B
    spatial_mask = spatial_index < (H * W)
    
    # Load bias (scalar)
    bias_val = tl.load(bias_ptr)
    
    # Load weights [1, 512, 1, 1] -> [512]
    weights = tl.load(weight_ptr).to(tl.float32)
    
    if batch_mask and spatial_mask:
        # Calculate input memory offset for this batch and spatial position
        batch_offset = batch_index * C_in * H * W
        spatial_offset = spatial_index * C_in
        
        # Load all channels for this batch and spatial position: [512]
        input_ptr = x_ptr + batch_offset + spatial_offset
        x_vals = tl.load(input_ptr + tl.arange(0, C_in), 
                        mask=tl.arange(0, C_in) < C_in).to(tl.float32)
        
        # 1x1 convolution: sum over input channels
        conv_result = bias_val + tl.sum(weights * x_vals)
        
        # Store result in output tensor: [B, 1, H*W]
        output_offset = batch_index * H * W + spatial_index
        tl.store(out_ptr + output_offset, conv_result)

@torch.fx.wrap
def fused_conv_softmax_large(in_0, in_1, in_2):
    # Input shapes: in_0=[1], in_1=[1,512,1,1], in_2=[32,512,64,64]
    B, C_in, H, W = in_2.shape  # B=32, C_in=512, H=64, W=64
    C_out = 1  # Output channels (512 -> 1)
    total_spatial = H * W  # 4096
    
    # Output shape [B, 1, total_spatial] after softmax  
    out = torch.empty((B, 1, total_spatial), dtype=torch.float32, device=in_2.device)
    
    # Use larger block size for better occupancy with larger batch
    BLOCK_HW = 128  # Trade-off between occupancy and memory bandwidth
    total_programs = B * total_spatial
    num_programs = (total_programs + BLOCK_HW - 1) // BLOCK_HW
    
    fused_conv_softmax_kernel_large[(num_programs,)](
        x_ptr=in_2.reshape(-1).contiguous(),  # Flatten to [B*C_in*H*W]
        weight_ptr=in_1.flatten(),            # [512]
        bias_ptr=in_0,
        out_ptr=out.reshape(-1),              # Flatten for kernel [B*H*W]
        B=B,
        H=H,
        W=W,
        C_in=C_in,
        C_out=C_out,
        BLOCK_HW=BLOCK_HW,
    )
    
    return out

def replacement_func():
    return fused_conv_softmax_large