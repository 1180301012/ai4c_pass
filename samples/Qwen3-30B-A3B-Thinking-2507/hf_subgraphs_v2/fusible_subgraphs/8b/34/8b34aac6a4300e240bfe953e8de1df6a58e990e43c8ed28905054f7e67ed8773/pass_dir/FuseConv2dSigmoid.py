import torch
import triton
import triton.language as tl

# Pattern matching function
@triton.jit
def conv1x1_sigmoid_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    B, H, W, C_in, C_out,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate global thread index
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    b_idx = idx // (H * W)
    h_idx = (idx % (H * W)) // W
    w_idx = (idx % (H * W)) % W

    # Mask for valid spatial indices
    b_mask = (b_idx < B)
    h_mask = (h_idx < H)
    w_mask = (w_idx < W)
    valid = b_mask & h_mask & w_mask

    # Load input for current spatial position (b_idx, h_idx, w_idx), shape (C_in,)
    input_offsets = b_idx * (H * W * C_in) + h_idx * (W * C_in) + w_idx * C_in + tl.arange(0, C_in)
    input_data = tl.load(input_ptr + input_offsets, mask=valid, other=0.0)

    # Load weight (C_out, C_in)
    c = tl.arange(0, C_out)
    weight_offsets = c[:, None] * C_in + tl.arange(0, C_in)
    weight_data = tl.load(weight_ptr + weight_offsets, mask=(c < C_out), other=0.0)

    # Compute dot product for all output channels
    output = tl.zeros((C_out,), dtype=tl.float32)
    for i in range(C_in):
        output += input_data[i] * weight_data[:, i]

    # Apply bias
    if bias_ptr is not None:
        bias_data = tl.load(bias_ptr + c, mask=(c < C_out), other=0.0)
        output += bias_data

    # Apply sigmoid
    output = 1 / (1 + tl.exp(-output))

    # Store output
    output_offsets = b_idx * (H * W * C_out) + h_idx * (W * C_out) + w_idx * C_out + c
    tl.store(output_ptr + output_offsets, output, mask=(c < C_out) & valid)

@torch.fx.wrap
def conv1x1_sigmoid(in_5, in_1, in_0):
    # Get input shapes
    B, C_in, H, W = in_5.shape
    _, C_out, _, _ = in_1.shape  # weight shape [C_out, C_in, 1, 1]
    C_out_bias = in_0.shape[0]  # bias shape [C_out]

    # Convert weight to 2D [C_out, C_in] by squeezing last two dims
    weight_2d = in_1.squeeze(2).squeeze(3)

    # Initialize output
    output = torch.empty((B, C_out, H, W), dtype=in_5.dtype, device=in_5.device)

    # Grid size: handle spatial elements (B*H*W)
    num_spatial = B * H * W
    BLOCK_SIZE = 256  # Block size for spatial dimension
    grid = (num_spatial + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch kernel
    conv1x1_sigmoid_kernel[grid](
        in_5, weight_2d, in_0, output,
        B, H, W, C_in, C_out,
        BLOCK_SIZE
    )
    return output

# Pattern matching function
def pattern(in_5, in_1, in_0):
    tmp_2 = torch.conv2d(in_5, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_6 = torch.sigmoid(tmp_2)
    return tmp_2, tmp_6

# Argument extraction function
def replacement_args(in_5, in_1, in_0):
    return (in_5, in_1, in_0)

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return conv1x1_sigmoid