import torch
import triton
import triton.language as tl

# Pattern matching function
# Matches: einsum('bchj,bhwj->bchw', in_4, in_1)
def pattern(in_4, in_1):
    return torch.functional.einsum('bchj,bhwj->bchw', in_4, in_1)

# Argument extraction function
# Extracts input tensors and tensor dimensions

def replacement_args(in_4, in_1):
    batch = in_4.shape[0]
    channels = in_4.shape[1]
    height = in_4.shape[2]
    width = in_1.shape[2]
    j_size = in_4.shape[3]
    return (in_4, in_1, batch, channels, height, width, j_size)

# Triton kernel for optimized einsum
@triton.jit
def einsum_kernel(
    in_4_ptr, in_1_ptr, out_ptr,
    batch, channels, height, width, j_size,
    BLOCK_C: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    # Get block indices
    c_block = tl.program_id(0) * BLOCK_C
    h_block = tl.program_id(1) * BLOCK_H
    w_block = tl.program_id(2) * BLOCK_W

    # Create index ranges for current block
    c = c_block + tl.arange(0, BLOCK_C)
    h = h_block + tl.arange(0, BLOCK_H)
    w = w_block + tl.arange(0, BLOCK_W)

    # Mask for valid indices
    mask_c = c < channels
    mask_h = h < height
    mask_w = w < width

    # Initialize accumulator to zero
    acc = tl.zeros((BLOCK_C, BLOCK_H, BLOCK_W), dtype=tl.float32)

    # Process j dimension in blocks
    for j_start in range(0, j_size, BLOCK_J):
        j_mask = j_start + tl.arange(0, BLOCK_J) < j_size
        j = j_start + tl.arange(0, BLOCK_J)

        # Load in_4: [c, h, j] -> (BLOCK_C, BLOCK_H, BLOCK_J)
        in_4 = tl.load(
            in_4_ptr + batch * channels * height * j_size + 
                      c[:, None] * height * j_size + 
                      h[:, None] * j_size + 
                      j,
            mask=mask_c[:, None] & mask_h[:, None] & j_mask,
            other=0.0
        )

        # Load in_1: [h, w, j] -> (BLOCK_H, BLOCK_W, BLOCK_J)
        in_1 = tl.load(
            in_1_ptr + batch * height * width * j_size + 
                      h[:, None] * width * j_size + 
                      w[None, :] * j_size + 
                      j,
            mask=mask_h[:, None] & mask_w[None, :] & j_mask,
            other=0.0
        )

        # Accumulate the product
        acc += in_4 * in_1

    # Store results
    out = (
        out_ptr + batch * channels * height * width +
        c[:, None, None] * height * width +
        h[:, None, None] * width +
        w[None, :, None]
    )
    tl.store(out, acc, 
             mask=mask_c[:, None, None] & mask_h[:, None, None] & mask_w[None, :, None])

# Kernel wrapper
@torch.fx.wrap
def triton_einsum(in_4, in_1):
    batch = in_4.shape[0]
    channels = in_4.shape[1]
    height = in_4.shape[2]
    width = in_1.shape[2]
    j_size = in_4.shape[3]

    out = torch.empty((batch, channels, height, width), 
                      dtype=in_4.dtype, device=in_4.device)

    # Block sizes tuned for NVIDIA GPUs
    BLOCK_C = 32
    BLOCK_H = 32
    BLOCK_W = 32
    BLOCK_J = 32

    num_blocks_c = (channels + BLOCK_C - 1) // BLOCK_C
    num_blocks_h = (height + BLOCK_H - 1) // BLOCK_H
    num_blocks_w = (width + BLOCK_W - 1) // BLOCK_W

    # Launch kernel
    einsum_kernel[(num_blocks_c, num_blocks_h, num_blocks_w), ](
        in_4, in_1, out,
        batch, channels, height, width, j_size,
        BLOCK_C=BLOCK_C, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, BLOCK_J=BLOCK_J
    )

    return out

# Replacement function

def replacement_func():
    return triton_einsum