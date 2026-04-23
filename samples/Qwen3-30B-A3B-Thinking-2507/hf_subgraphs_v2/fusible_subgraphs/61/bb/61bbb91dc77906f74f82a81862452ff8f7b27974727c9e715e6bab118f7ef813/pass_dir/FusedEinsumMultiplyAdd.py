import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3, in_4):
    einsum = torch.functional.einsum('bchj,bhwj->bchw', in_4, in_1)
    in_3 += einsum
    in_5 = in_3
    res = in_5 * in_0 + in_2
    return res

def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)

@triton.jit
def fused_einsum_kernel(
    in_4_ptr, in_1_ptr, in_3_ptr, in_2_ptr, out_ptr,
    batch_size, channels, height, width, j_dim,
    in_0_scalar,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr
):
    # Calculate global indices
    h_block = tl.program_id(0)
    w_block = tl.program_id(1)
    batch_idx = tl.program_id(2)
    channel_idx = tl.program_id(3)

    # Compute starting h and w for current block
    h_start = h_block * BLOCK_H
    w_start = w_block * BLOCK_W

    # Initialize accumulators
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    # Load and compute reduction over j dimension
    for j in range(0, j_dim, 32):
        j_end = min(j + 32, j_dim)
        
        # Load in_4: [batch, channel, h, j] -> [BLOCK_H, j_end-j]
        in_4_block = tl.load(
            in_4_ptr + 
            (batch_idx * channels * height * j_dim + 
             channel_idx * height * j_dim + 
             h_start * j_dim + j),
            shape=(BLOCK_H, j_end - j),
            mask=tl.arange(0, BLOCK_H)[:, None] < height - h_start,
            other=0.0
        )

        # Load in_1: [batch, h, w, j] -> [BLOCK_H, j_end-j]
        in_1_block = tl.load(
            in_1_ptr + 
            (batch_idx * height * width * j_dim + 
             h_start * width * j_dim + 
             w_start * j_dim + j),
            shape=(BLOCK_H, j_end - j),
            mask=tl.arange(0, BLOCK_H)[:, None] < height - h_start,
            other=0.0
        )

        # Compute inner product
        acc += tl.dot(in_4_block, in_1_block.T)

    # Apply remaining operations
    for h in range(BLOCK_H):
        for w in range(BLOCK_W):
            h_idx = h_start + h
            w_idx = w_start + w
            if h_idx < height and w_idx < width:
                # Load in_3 and in_2
                in_3_val = tl.load(
                    in_3_ptr + batch_idx * channels * height * width + 
                    channel_idx * height * width + h_idx * width + w_idx
                )
                in_2_val = tl.load(
                    in_2_ptr + batch_idx * channels * height * width + 
                    channel_idx * height * width + h_idx * width + w_idx
                )
                
                # Compute result
                out_val = (acc[h, w] + in_3_val) * in_0_scalar + in_2_val
                
                # Store to output
                tl.store(
                    out_ptr + batch_idx * channels * height * width + 
                    channel_idx * height * width + h_idx * width + w_idx,
                    out_val
                )

@torch.fx.wrap
def kernel_wrapper(in_0, in_1, in_2, in_3, in_4):
    batch_size = in_3.shape[0]
    channels = in_3.shape[1]
    height = in_3.shape[2]
    width = in_3.shape[3]
    j_dim = in_1.shape[3]
    in_0_scalar = in_0.item()

    # Compute grid dimensions
    num_h_blocks = (height + 32 - 1) // 32
    num_w_blocks = (width + 32 - 1) // 32

    out = torch.empty_like(in_3)

    # Launch kernel with optimal block sizes
    fused_einsum_kernel[(num_h_blocks, num_w_blocks, batch_size, channels)](
        in_4, in_1, in_3, in_2, out,
        batch_size, channels, height, width, j_dim,
        in_0_scalar,
        32, 32
    )

    return out

def replacement_func():
    return kernel_wrapper