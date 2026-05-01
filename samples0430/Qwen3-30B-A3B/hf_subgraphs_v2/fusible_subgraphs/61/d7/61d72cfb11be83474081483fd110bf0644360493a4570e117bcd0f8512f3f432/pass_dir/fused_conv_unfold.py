import torch
import triton
import triton.language as tl

# Pattern matching function

def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.nn.functional.unfold(conv2d, kernel_size=(2, 2), stride=(2, 2))
    tmp_3 = tmp_2.reshape(1, 128, 4, -1)
    return tmp_3

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_conv_unfold_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    num_blocks,
    BLOCK_SIZE: tl.constexpr
):
    # Each block processes one spatial block (i, j) in output
    block_id = tl.program_id(0)
    i = block_id // 16
    j = block_id % 16
    
    # Calculate spatial positions for 2x2 block
    h0 = 2 * i
    w0 = 2 * j
    h1 = h0
    w1 = w0 + 1
    h2 = h0 + 1
    w2 = w0
    h3 = h0 + 1
    w3 = w0 + 1

    # Get channel index from thread
    thread_c = tl.thread_id(0)
    
    # Early exit for threads beyond channel count
    if thread_c >= 128:
        return

    # Initialize accumulators
    val0 = tl.zeros([1], dtype=tl.float32)
    val1 = tl.zeros([1], dtype=tl.float32)
    val2 = tl.zeros([1], dtype=tl.float32)
    val3 = tl.zeros([1], dtype=tl.float32)

    # Process all input channels for the 2x2 window
    for k in tl.arange(0, 256):
        # Load input for (h0, w0)
        input_val = tl.load(
            input_ptr + k * 1024 + h0 * 32 + w0
        )
        weight_val = tl.load(weight_ptr + thread_c * 256 + k)
        val0 += input_val * weight_val

        # Load input for (h1, w1)
        input_val = tl.load(
            input_ptr + k * 1024 + h1 * 32 + w1
        )
        val1 += input_val * weight_val

        # Load input for (h2, w2)
        input_val = tl.load(
            input_ptr + k * 1024 + h2 * 32 + w2
        )
        val2 += input_val * weight_val

        # Load input for (h3, w3)
        input_val = tl.load(
            input_ptr + k * 1024 + h3 * 32 + w3
        )
        val3 += input_val * weight_val

    # Store results in output
    base_ptr = output_ptr + thread_c * 1024 + block_id * 4
    tl.store(base_ptr + 0, val0)
    tl.store(base_ptr + 1, val1)
    tl.store(base_ptr + 2, val2)
    tl.store(base_ptr + 3, val3)


@torch.fx.wrap
def kernel_wrapper(in_0, in_1):
    # Reshape weight to [128, 256] for the kernel
    weight = in_0.reshape(128, 256)
    
    # Input shape: [1, 256, 32, 32]
    _, _, H, W = in_1.shape
    num_blocks = (H // 2) * (W // 2)  # 16*16 = 256
    output = torch.empty(
        1, 128, 4, num_blocks,
        device=in_1.device,
        dtype=in_1.dtype
    )

    # Grid dimensions: 256 blocks (spatial blocks), 128 threads/block (channels)
    grid = (num_blocks,)
    fused_conv_unfold_kernel[grid](
        in_1,
        weight,
        output,
        num_blocks,
        BLOCK_SIZE=128
    )
    
    return output


def replacement_func():
    return kernel_wrapper