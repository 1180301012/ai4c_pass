import torch
import triton
import triton.language as tl

# Constants
STRIDE_T = 2
STRIDE_H = 16
STRIDE_W = 16
KERNEL_T = 2
KERNEL_H = 16
KERNEL_W = 16

# Pattern matching function
# Matches the convolution, flatten, and transpose sequence
def pattern(in_0, in_1, in_3):
    conv3d = torch.conv3d(in_3, in_1, in_0, (STRIDE_T, STRIDE_H, STRIDE_W), (0, 0, 0), (1, 1, 1), 1)
    tmp_4 = conv3d.flatten(2)
    tmp_5 = tmp_4.transpose(1, 2)
    return tmp_5

# Argument extraction function
# Extracts the necessary tensors for the kernel
@torch.fx.wrap
def replacement_args(in_0, in_1, in_3):
    return (in_0, in_1, in_3)

# Triton kernel for 3D convolution + flatten + transpose
@triton.jit
def conv3d_flatten_transpose_kernel(
    input_ptr, weight_ptr, bias_ptr,
    output_ptr,
    batch, in_c, T, H, W,
    out_c, T_out, H_out, W_out,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate global index
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    total_elements = batch * T_out * H_out * W_out * out_c

    # Process elements in this block
    for i in range(start_idx, min(start_idx + BLOCK_SIZE, total_elements)):
        # Convert 1D index to 3D spatial and channel indices
        batch_idx = i // (T_out * H_out * W_out * out_c)
        remainder = i % (T_out * H_out * W_out * out_c)
        patch_idx = remainder // out_c
        out_c_idx = remainder % out_c

        # Convert patch index to spatial coordinates
        t = patch_idx // (H_out * W_out)
        h = (patch_idx % (H_out * W_out)) // W_out
        w = patch_idx % W_out

        # Calculate input window start
        input_t_start = t * STRIDE_T
        input_h_start = h * STRIDE_H
        input_w_start = w * STRIDE_W

        # Accumulate convolution result
        acc = tl.zeros((1,), dtype=tl.float32)
        for c in range(in_c):
            for kt in range(KERNEL_T):
                for kh in range(KERNEL_H):
                    for kw in range(KERNEL_W):
                        # Input access
                        input_t = input_t_start + kt
                        input_h = input_h_start + kh
                        input_w = input_w_start + kw
                        input_idx = (batch_idx * in_c * T * H * W) + \
                                    (c * T * H * W) + \
                                    (input_t * H * W) + \
                                    (input_h * W) + \
                                    input_w
                        input_val = tl.load(input_ptr + input_idx, \
                                          mask=input_t < T and input_h < H and input_w < W, \
                                          other=0.0)

                        # Weight access
                        weight_idx = (out_c_idx * in_c * KERNEL_T * KERNEL_H * KERNEL_W) + \
                                    (c * KERNEL_T * KERNEL_H * KERNEL_W) + \
                                    (kt * KERNEL_H * KERNEL_W) + \
                                    (kh * KERNEL_W) + \
                                    kw
                        weight_val = tl.load(weight_ptr + weight_idx)

                        acc += input_val * weight_val

        # Add bias
        bias_val = tl.load(bias_ptr + out_c_idx)
        acc += bias_val

        # Store result in output tensor (converted to float32)
        tl.store(output_ptr + i, acc)

# Kernel wrapper function
@torch.fx.wrap
def conv3d_flatten_transpose(in_0, in_1, in_3):
    # Handle input dtype conversion to float32 for computation
    input_float32 = in_3.to(torch.float32)
    weight_float32 = in_1.to(torch.float32)
    bias_float32 = in_0.to(torch.float32)

    # Get input dimensions
    batch = input_float32.shape[0]
    in_c = input_float32.shape[1]
    T = input_float32.shape[2]
    H = input_float32.shape[3]
    W = input_float32.shape[4]

    # Compute output spatial dimensions
    T_out = (T - KERNEL_T) // STRIDE_T + 1
    H_out = (H - KERNEL_H) // STRIDE_H + 1
    W_out = (W - KERNEL_W) // STRIDE_W + 1
    total_patches = T_out * H_out * W_out
    out_c = weight_float32.shape[0]

    # Allocate output tensor in float32
    output_float32 = torch.empty((batch, total_patches, out_c), \
                                dtype=torch.float32, device=input_float32.device)

    # Configure kernel grid
    num_elements = batch * total_patches * out_c
    BLOCK_SIZE = 128
    grid = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch kernel
    conv3d_flatten_transpose_kernel[grid](
        input_float32, weight_float32, bias_float32,
        output_float32,
        batch, in_c, T, H, W,
        out_c, T_out, H_out, W_out,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Convert back to original input dtype
    return output_float32.to(in_3.dtype)

# Replacement function (returns kernel wrapper)
def replacement_func():
    return conv3d_flatten_transpose