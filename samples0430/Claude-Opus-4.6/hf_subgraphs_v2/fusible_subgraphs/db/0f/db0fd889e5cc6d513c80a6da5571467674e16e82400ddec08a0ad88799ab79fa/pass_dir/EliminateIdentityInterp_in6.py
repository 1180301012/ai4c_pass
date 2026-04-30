import torch
import triton
import triton.language as tl


def pattern(in_6):
    tmp_25 = in_6[(slice(None, None, None), slice(None, None, None), 0, slice(None, None, None))]
    tmp_26 = tmp_25[(slice(None, None, None), None)]
    tmp_27 = in_6[(slice(None, None, None), slice(None, None, None), slice(-10, None, None), slice(None, None, None))]
    tmp_28 = in_6[(slice(None, None, None), slice(None, None, None), slice(1, -10, None), slice(None, None, None))]
    tmp_29 = tmp_28.transpose(2, 3)
    tmp_30 = tmp_29.view(4, 32, 15, 15)
    tmp_31 = torch.nn.functional.interpolate(tmp_30, size = (15, 15), mode = 'bicubic', align_corners = False)
    tmp_32 = tmp_31.flatten(2)
    tmp_33 = tmp_32.transpose(1, 2)
    tmp_34 = tmp_33.contiguous()
    tmp_35 = tmp_34.view(4, 1, 225, 32)
    return (tmp_26, tmp_27, tmp_35)


def replacement_args(in_6):
    return (in_6,)


@triton.jit
def slice_copy_kernel(src_ptr, dst_ptr,
                      batch_size, copy_rows, cols,
                      src_batch_stride, src_row_stride,
                      start_row,
                      n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Decompose linear dst index to (batch, row, col)
    col = offsets % cols
    remainder = offsets // cols
    row = remainder % copy_rows
    batch = remainder // copy_rows

    # Source offset: element at (batch, 0, start_row+row, col) in the 4D tensor
    src_off = batch * src_batch_stride + (start_row + row) * src_row_stride + col

    data = tl.load(src_ptr + src_off, mask=mask)
    tl.store(dst_ptr + offsets, data, mask=mask)


@torch.fx.wrap
def process_in6(in_6):
    # in_6: [4, 1, 236, 32], contiguous parameter
    # The interpolation from (15,15) to (15,15) is identity.
    # Net result:
    #   tmp_26 = in_6[:, :, 0:1, :] -> [4, 1, 1, 32]
    #   tmp_27 = in_6[:, :, -10:, :] -> [4, 1, 10, 32]
    #   tmp_35 = in_6[:, :, 1:-10, :].contiguous() -> [4, 1, 225, 32]

    src_batch_stride = in_6.stride(0)
    src_row_stride = in_6.stride(2)
    cols = 32
    batch_size = 4

    out1 = torch.empty([4, 1, 1, 32], dtype=in_6.dtype, device=in_6.device)
    out2 = torch.empty([4, 1, 10, 32], dtype=in_6.dtype, device=in_6.device)
    out3 = torch.empty([4, 1, 225, 32], dtype=in_6.dtype, device=in_6.device)

    BLOCK_SIZE = 1024

    # Copy out1: row 0 (tmp_26 = in_6[:,:,0:1,:])
    n1 = 4 * 1 * 32
    grid1 = ((n1 + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    slice_copy_kernel[grid1](in_6, out1, batch_size, 1, cols,
                             src_batch_stride, src_row_stride, 0, n1,
                             BLOCK_SIZE=BLOCK_SIZE)

    # Copy out2: rows 226:236 (tmp_27 = in_6[:,:,-10:,:])
    n2 = 4 * 10 * 32
    grid2 = ((n2 + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    slice_copy_kernel[grid2](in_6, out2, batch_size, 10, cols,
                             src_batch_stride, src_row_stride, 226, n2,
                             BLOCK_SIZE=BLOCK_SIZE)

    # Copy out3: rows 1:226 (tmp_35 = in_6[:,:,1:-10,:].contiguous())
    n3 = 4 * 225 * 32
    grid3 = ((n3 + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    slice_copy_kernel[grid3](in_6, out3, batch_size, 225, cols,
                             src_batch_stride, src_row_stride, 1, n3,
                             BLOCK_SIZE=BLOCK_SIZE)

    return (out1, out2, out3)


def replacement_func():
    return process_in6