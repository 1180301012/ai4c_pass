import torch
import triton
import triton.language as tl

# Pattern matching function - matches the exact computation graph
def pattern(in_0, in_1):
    # conv2d: input (1, 256, 32, 32) @ weight (128, 256, 1, 1) -> output (1, 128, 32, 32)
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    # unfold: extract 2x2 patches with stride 2 -> output (1, 128*4, 16*16) = (1, 512, 256)
    tmp_2 = torch.nn.functional.unfold(conv2d, kernel_size=(2, 2), stride=(2, 2))
    # reshape: -> output (1, 128, 4, 256)
    tmp_3 = tmp_2.reshape(1, 128, 4, -1)
    return conv2d, tmp_2, tmp_3

# Extract arguments from matched pattern
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized Triton kernel for fused conv2d + unfold + reshape
@triton.jit
def fused_conv_unfold_reshape_kernel(
    weight_ptr,
    input_ptr,
    output_ptr,
    # Tensor dimensions
    batch_size, in_channels, out_channels,
    input_height, input_width,
    # Weight is (out_channels, in_channels, 1, 1) -> 128 x 256
    # Output shape: (batch, out_channels, 4, spatial_patches) where spatial_patches = (H/2)*(W/2)
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a portion of the output
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Output shape: (batch=1, out_channels=128, 4, spatial_patches=256)
    # Linearize: idx = (((batch * out_channels + oc) * 4 + patch_idx) * spatial_size + spatial_offset)
    # For batch=0: idx = oc * 1024 + patch_idx * 256 + spatial_offset
    
    batch = 0
    out_ch = offsets // 1024
    tmp = offsets % 1024
    patch_idx = tmp // 256  # 0, 1, 2, 3
    spatial_offset = tmp % 256  # 0-255
    
    # Extract 2x2 patch coordinates based on spatial_offset and patch_idx
    # spatial_offset ranges from 0 to 255 representing (16x16) spatial positions
    # patch_idx determines which of the 4 patches (0,1,2,3) -> top-left, top-right, bottom-left, bottom-right
    h_base = (spatial_offset // 16) * 2
    w_base = (spatial_offset % 16) * 2
    
    # patch_idx: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
    h_offset = (patch_idx // 2) * 1
    w_offset = (patch_idx % 2) * 1
    
    h = h_base + h_offset
    w = w_base + w_offset
    
    # Compute conv2d + unfold in one pass
    # conv2d with 1x1 kernel is: out[b, c_out] = sum_c_in(in[b, c_in] * weight[c_out, c_in])
    # For each output channel, accumulate over all input channels
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Vectorized load for input patch (2x2 = 4 elements)
    # Input coords for the 2x2 patch
    in_h0 = h
    in_w0 = w
    in_h1 = h + 1  # for 2x2 patch
    in_w1 = w + 1
    
    # Linear input indices for the 2x2 patch
    # input is (batch=1, channels, H, W) = (1, 256, 32, 32)
    # idx = ((batch * in_channels + c) * H + h) * W + w
    c = tl.arange(0, in_channels)  # 256 channels
    
    # Load 4 input values for the 2x2 patch
    idx_base = h * input_width + w
    idx_h1 = (h + 1) * input_width + w
    idx_h0_w1 = h * input_width + (w + 1)
    idx_h1_w1 = (h + 1) * input_width + (w + 1)
    
    # Load input values (bfloat16/float16 need conversion)
    inp0 = tl.load(input_ptr + idx_base * in_channels + c, mask=c < in_channels, other=0.0)
    inp1 = tl.load(input_ptr + idx_h1 * in_channels + c, mask=c < in_channels, other=0.0)
    inp2 = tl.load(input_ptr + idx_h0_w1 * in_channels + c, mask=c < in_channels, other=0.0)
    inp3 = tl.load(input_ptr + idx_h1_w1 * in_channels + c, mask=c < in_channels, other=0.0)
    
    # Load weight values for this output channel
    # weight is (out_channels, in_channels, 1, 1)
    # For 1x1 conv, we just use weight[oc, c, 0, 0]
    weight_oc = out_ch
    wgt = tl.load(weight_ptr + weight_oc * in_channels + c, mask=c < in_channels, other=0.0)
    
    # Compute dot product: sum over input channels
    # Patch element 0: inp0 * weight
    # Patch element 1: inp1 * weight
    # Patch element 2: inp2 * weight
    # Patch element 3: inp3 * weight
    acc0 = tl.sum(inp0 * wgt)
    acc1 = tl.sum(inp1 * wgt)
    acc2 = tl.sum(inp2 * wgt)
    acc3 = tl.sum(inp3 * wgt)
    
    # Store all 4 patch elements
    # Each output element has its own position
    out_base = out_ch * 1024 + patch_idx * 256 + spatial_offset
    
    # Compute actual output indices for each patch element
    tl.store(output_ptr + out_base, acc0.to(tl.float16), mask=mask)
    tl.store(output_ptr + out_base + 256, acc1.to(tl.float16), mask=(offsets + 256) < n_elements)
    tl.store(output_ptr + out_base + 512, acc2.to(tl.float16), mask=(offsets + 512) < n_elements)
    tl.store(output_ptr + out_base + 768, acc3.to(tl.float16), mask=(offsets + 768) < n_elements)


@torch.fx.wrap
def fused_conv_unfold_reshape(weight, input):
    """
    Fused conv2d + unfold + reshape kernel.
    
    Input shape: (1, 256, 32, 32)
    Weight shape: (128, 256, 1, 1)
    Output shape: (1, 128, 4, 256)
    """
    batch_size = 1
    in_channels = 256
    out_channels = 128
    input_height = 32
    input_width = 32
    
    # Output has (batch * out_channels * 4 * spatial_patches) = 1 * 128 * 4 * 256 = 131072 elements
    # But we process 4 elements at a time (one patch), so we need 128 * 256 = 32768 launches
    # Actually: each thread writes 4 values, so we need (128 * 256) = 32768 threads
    n_elements = out_channels * 256  # spatial patches = 256
    
    BLOCK_SIZE = 256
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with same dtype as input
    output = torch.empty((batch_size, out_channels, 4, 256), dtype=input.dtype, device=input.device)
    
    # Launch kernel
    fused_conv_unfold_reshape_kernel[(num_programs,)](
        weight_ptr=weight,
        input_ptr=input,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        input_height=input_height,
        input_width=input_width,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return fused_conv_unfold_reshape