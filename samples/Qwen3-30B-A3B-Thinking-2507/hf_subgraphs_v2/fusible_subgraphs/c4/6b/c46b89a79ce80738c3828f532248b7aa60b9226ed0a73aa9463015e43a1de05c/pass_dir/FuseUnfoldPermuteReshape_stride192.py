import torch
import triton
import triton.language as tl

def pattern(in_1):
    tmp_0 = torch.nn.functional.unfold(in_1, kernel_size = (384, 384), stride = (192, 192));  in_1 = None
    tmp_1 = tmp_0.permute(2, 0, 1);  tmp_0 = None
    tmp_2 = tmp_1.reshape(-1, 3, 384, 384);  tmp_1 = None
    return tmp_2

def replacement_args(in_1):
    return (in_1,)

@triton.jit
def patch_extract_kernel(
    input_ptr,
    output_ptr,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    H_k: tl.constexpr,
    W_k: tl.constexpr,
    H_s: tl.constexpr,
    W_s: tl.constexpr,
    n_patches: tl.constexpr,
    block_size_r: tl.constexpr,
    block_size_s: tl.constexpr,
):
    k = tl.program_id(0)
    i = k // 3
    j = k % 3
    start_h = i * H_s
    start_w = j * W_s

    r = tl.program_id(1) * block_size_r + tl.thread_id(0)
    s = tl.program_id(2) * block_size_s + tl.thread_id(1)
    c = tl.thread_id(2)

    if r < H_k and s < W_k and c < 3:
        input_offset = c * (H_in * W_in) + (start_h + r) * W_in + (start_w + s)
        input_val = tl.load(input_ptr + input_offset)
        output_offset = k * (3 * H_k * W_k) + c * (H_k * W_k) + r * W_k + s
        tl.store(output_ptr + output_offset, input_val)

@torch.fx.wrap
def optimized_unfold(in_1):
    H_in, W_in = in_1.shape[2], in_1.shape[3]
    H_k, W_k = 384, 384
    H_s, W_s = 192, 192
    C = 3

    H_patches = (H_in - H_k) // H_s + 1
    W_patches = (W_in - W_k) // W_s + 1
    n_patches = H_patches * W_patches

    output = torch.empty((n_patches, C, H_k, W_k), dtype=in_1.dtype, device=in_1.device)
    input_ptr = in_1.data_ptr()
    output_ptr = output.data_ptr()

    block_size_r = 16
    block_size_s = 16

    grid_x = n_patches
    grid_y = (H_k + block_size_r - 1) // block_size_r
    grid_z = (W_k + block_size_s - 1) // block_size_s

    patch_extract_kernel[(grid_x, grid_y, grid_z)](
        input_ptr=input_ptr,
        output_ptr=output_ptr,
        H_in=H_in,
        W_in=W_in,
        H_k=H_k,
        W_k=W_k,
        H_s=H_s,
        W_s=W_s,
        n_patches=n_patches,
        block_size_r=block_size_r,
        block_size_s=block_size_s,
    )

    return output

def replacement_func():
    return optimized_unfold