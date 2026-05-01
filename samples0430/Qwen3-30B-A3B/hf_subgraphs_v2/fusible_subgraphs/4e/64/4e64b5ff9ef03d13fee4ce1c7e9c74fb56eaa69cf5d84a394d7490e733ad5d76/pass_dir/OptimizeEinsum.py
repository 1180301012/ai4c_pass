import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_2, in_1):
    einsum = torch.functional.einsum('bchw,bchj->bhwj', in_2, in_1)
    return einsum

# Argument extraction function
def replacement_args(in_2, in_1):
    return (in_2, in_1)


# Triton kernel implementation for einsum operation
@triton.jit
def einsum_kernel(
    in_2_ptr,
    in_1_ptr,
    out_ptr,
    B: tl.int32,
    H: tl.int32,
    C: tl.int32,
    W: tl.int32,
    J: tl.int32,
    BLOCK_W: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    # Calculate global block ID
    block_id = tl.program_id(0)
    # Decompose block ID to batch and tile positions
    num_w_tiles = (W + BLOCK_W - 1) // BLOCK_W
    num_j_tiles = (J + BLOCK_J - 1) // BLOCK_J
    batch_idx = block_id // (num_w_tiles * num_j_tiles)
    w_tile = (block_id % (num_w_tiles * num_j_tiles)) // num_j_tiles
    j_tile = block_id % num_j_tiles
    
    w_start = w_tile * BLOCK_W
    j_start = j_tile * BLOCK_J

    # Calculate batch coordinates (b, h)
    b = batch_idx // H
    h = batch_idx % H

    # Calculate base pointers for current batch
    in_2_batch_ptr = in_2_ptr + b * C * H * W + h * C * W
    in_1_batch_ptr = in_1_ptr + b * C * H * J + h * C * J

    # Each thread handles one (w, j) within tile
    w_idx = w_start + (tl.thread_id(0) % BLOCK_W)
    j_idx = j_start + (tl.thread_id(0) // BLOCK_W)

    # Mask for boundary checking
    w_mask = w_idx < W
    j_mask = j_idx < J
    
    # Initialize accumulator
    acc = 0.0
    # Sum over C dimension
    for c in range(C):
        in_2_val = tl.load(in_2_batch_ptr + c * W + w_idx, mask=w_mask, other=0.0)
        in_1_val = tl.load(in_1_batch_ptr + c * J + j_idx, mask=j_mask, other=0.0)
        acc += in_2_val * in_1_val

    # Calculate output pointer offset
    out_offset = b * H * W * J + h * W * J + w_idx * J + j_idx
    tl.store(out_ptr + out_offset, acc, mask=w_mask & j_mask)


# Kernel wrapper
@torch.fx.wrap
def optimized_einsum(in_2, in_1):
    B, C, H, W = in_2.shape
    _, _, _, J = in_1.shape
    out = torch.empty((B, H, W, J), dtype=in_2.dtype, device=in_2.device)
    BLOCK_W, BLOCK_J = 16, 16
    
    # Calculate grid dimensions
    num_w_tiles = (W + BLOCK_W - 1) // BLOCK_W
    num_j_tiles = (J + BLOCK_J - 1) // BLOCK_J
    grid_size = B * H * num_w_tiles * num_j_tiles
    
    # Launch kernel
    einsum_kernel[grid_size](
        in_2,
        in_1,
        out,
        B,
        H,
        C,
        W,
        J,
        BLOCK_W,
        BLOCK_J
    )
    return out


# Replacement function (must return the kernel wrapper)
def replacement_func():
    return optimized_einsum