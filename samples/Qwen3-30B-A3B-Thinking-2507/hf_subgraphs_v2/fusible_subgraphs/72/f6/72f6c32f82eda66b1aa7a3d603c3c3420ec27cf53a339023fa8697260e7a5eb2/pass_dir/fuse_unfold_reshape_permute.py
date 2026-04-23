import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.nn.functional.pad(conv2d, [2, 2, 2, 2], 'constant', None)
    tmp_3 = tmp_2.unfold(2, 12, 8)
    tmp_4 = tmp_3.unfold(3, 12, 8)
    tmp_5 = tmp_4.reshape(8, 80, 4, -1)
    tmp_6 = tmp_5.permute(0, 2, 3, 1)
    split = torch.functional.split(tmp_6, [16, 64], dim=-1)
    tmp_8 = split[0]
    tmp_9 = split[1]
    tmp_10 = tmp_8.transpose(-1, -2)
    return (tmp_10, tmp_9)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fusion_kernel(
    in_ptr,
    out1_ptr,
    out2_ptr,
    batch_size,
    channels,
    h_in,
    w_in,
    h_blocks,
    w_blocks,
    block_size,
    stride,
    h_out,
    w_out,
    N_out1,
    N_out2,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_out1

    # Process out1 [8,4,16,144]
    for idx in offsets[mask]:
        i = idx // (4 * 16 * 144)
        j = (idx // (16 * 144)) % 4
        k = (idx // 144) % 16
        l = idx % 144

        # Map to unfolded tensor indices
        ch = idx // (h_blocks * w_blocks * block_size * block_size)
        h_block = (idx // (w_blocks * block_size * block_size)) % h_blocks
        w_block = (idx // (block_size * block_size)) % w_blocks
        h_offset = (idx // block_size) % block_size
        w_offset = idx % block_size

        # Calculate input spatial coordinates
        input_h = h_block * stride + h_offset
        input_w = w_block * stride + w_offset

        # Load from input tensor
        input_val = tl.load(
            in_ptr + ch * h_in * w_in + input_h * w_in + input_w,
            mask=(input_h < h_in) & (input_w < w_in),
            other=0.0
        )

        # Store to out1 (transpose applied directly)
        out1_idx = i * (4 * 16 * 144) + j * (16 * 144) + k * 144 + l
        tl.store(out1_ptr + out1_idx, input_val)

    # Process out2 [8,4,144,64] (similar logic)
    mask2 = offsets >= N_out1
    for idx in offsets[mask2]:
        out2_idx = idx - N_out1
        i = out2_idx // (4 * 144 * 64)
        j = (out2_idx // (144 * 64)) % 4
        k = (out2_idx // 64) % 144
        l = out2_idx % 64

        # Map to unfolded tensor indices
        ch = out2_idx // (h_blocks * w_blocks * block_size * block_size)
        h_block = (out2_idx // (w_blocks * block_size * block_size)) % h_blocks
        w_block = (out2_idx // (block_size * block_size)) % w_blocks
        h_offset = (out2_idx // block_size) % block_size
        w_offset = out2_idx % block_size

        # Calculate input spatial coordinates
        input_h = h_block * stride + h_offset
        input_w = w_block * stride + w_offset

        # Load from input tensor
        input_val = tl.load(
            in_ptr + ch * h_in * w_in + input_h * w_in + input_w,
            mask=(input_h < h_in) & (input_w < w_in),
            other=0.0
        )

        # Store to out2 (no transpose needed)
        out2_idx_final = i * (4 * 144 * 64) + j * (144 * 64) + k * 64 + l
        tl.store(out2_ptr + out2_idx_final, input_val)

@torch.fx.wrap
def kernel_wrapper(in_0, in_1):
    # Conv2d + pad already done in model, so we're working on the padded tensor
    padded = torch.nn.functional.pad(
        torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1),
        [2, 2, 2, 2], 'constant', None
    )

    # Input tensor shape: [1, 640, 20, 20]
    batch_size = 1
    channels = 640
    h_in = 20
    w_in = 20

    # Unfold params
    h_blocks = 2  # (20 - 12) // 8 + 1
    w_blocks = 2
    block_size = 12
    stride = 8
    h_out = (h_in - block_size) // stride + 1
    w_out = (w_in - block_size) // stride + 1

    # Output shapes
    out1_shape = (8, 4, 16, 144)
    out2_shape = (8, 4, 144, 64)
    N_out1 = 8 * 4 * 16 * 144
    N_out2 = 8 * 4 * 144 * 64

    # Allocate outputs
    out1 = torch.empty(out1_shape, dtype=padded.dtype, device=padded.device)
    out2 = torch.empty(out2_shape, dtype=padded.dtype, device=padded.device)

    # Launch kernel
    grid = (max(N_out1, N_out2) + 1023) // 1024
    fusion_kernel[(grid,),](
        padded,
        out1,
        out2,
        batch_size,
        channels,
        h_in,
        w_in,
        h_blocks,
        w_blocks,
        block_size,
        stride,
        h_out,
        w_out,
        N_out1,
        N_out2,
        BLOCK_SIZE=1024
    )
    
    return (out1, out2)

def replacement_func():
    return kernel_wrapper