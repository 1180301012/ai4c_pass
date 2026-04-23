import torch
import triton
import triton.language as tl

def pattern(tmp_2):
    return torch.nn.functional.interpolate(tmp_2, (64, 128), None, 'bilinear', False)

def replacement_args(tmp_2):
    return (tmp_2,)

@triton.jit
def bilinear_interp_kernel(
    input_ptr,
    output_ptr,
    C,
    BLOCK_SIZE: tl.constexpr = 128
):
    total_elements = 64 * 128 * C
    block_id = tl.program_id(0)
    start_idx = block_id * BLOCK_SIZE
    # Unpack idx into channel, height, width
    c = start_idx // (64 * 128)
    remainder = start_idx % (64 * 128)
    h = remainder // 128
    w = remainder % 128
    # Calculate source width coordinate
    source_w = w / 32.0  # 128/4 = 32
    i_w = tl.floor(source_w).to(tl.int32)
    f_w = source_w - i_w
    i_w_next = tl.minimum(i_w + 1, 3)
    # Load source values
    input_cw = tl.load(input_ptr + c * 4 + i_w)
    input_cw_next = tl.load(input_ptr + c * 4 + i_w_next)
    # Interpolate
    interpolated = input_cw * (1 - f_w) + input_cw_next * f_w
    # Store result
    output_offset = c * 64 * 128 + h * 128 + w
    tl.store(output_ptr + output_offset, interpolated)

@torch.fx.wrap
def bilinear_interp_wrapper(input_tensor):
    B, C, H_in, W_in = input_tensor.shape
    assert B == 1 and H_in == 1 and W_in == 4, "Expected input shape [1, C, 1, 4]"
    H_out = 64
    W_out = 128
    output = torch.empty((B, C, H_out, W_out), dtype=input_tensor.dtype, device=input_tensor.device)
    num_elements = B * C * H_out * W_out
    BLOCK_SIZE = 128
    num_blocks = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    bilinear_interp_kernel[(num_blocks,)](
        input_tensor,
        output,
        C,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output

def replacement_func():
    return bilinear_interp_wrapper