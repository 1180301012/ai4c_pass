import torch
import triton
import triton.language as tl


# Simple copy kernel - one for each input tensor
@triton.jit
def copy_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    data = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + offsets, data, mask=mask)


@torch.fx.wrap
def triton_cat(in_5, in_7, in_8, in_6, tmp_7):
    N, C0, H, W = in_5.shape
    C1 = in_7.shape[1]
    C2 = in_8.shape[1]
    C3 = in_6.shape[1]
    C4 = tmp_7.shape[1]
    
    HW = H * W
    total_channels = C0 + C1 + C2 + C3 + C4
    
    # Allocate output
    out = torch.empty((N, total_channels, H, W), dtype=in_5.dtype, device=in_5.device)
    
    BLOCK_SIZE = 1024
    
    # Copy each tensor to its position in the output
    # in_5 goes to channels [0, C0)
    n0 = C0 * HW
    copy_kernel[((n0 + BLOCK_SIZE - 1) // BLOCK_SIZE,)](
        src_ptr=in_5, dst_ptr=out, n_elements=n0, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # in_7 goes to channels [C0, C0+C1)
    n1 = C1 * HW
    dst_offset_1 = C0 * HW
    copy_kernel[((n1 + BLOCK_SIZE - 1) // BLOCK_SIZE,)](
        src_ptr=in_7, dst_ptr=out.view(-1)[dst_offset_1:], n_elements=n1, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # in_8 goes to channels [C0+C1, C0+C1+C2)
    n2 = C2 * HW
    dst_offset_2 = (C0 + C1) * HW
    copy_kernel[((n2 + BLOCK_SIZE - 1) // BLOCK_SIZE,)](
        src_ptr=in_8, dst_ptr=out.view(-1)[dst_offset_2:], n_elements=n2, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # in_6 goes to channels [C0+C1+C2, C0+C1+C2+C3)
    n3 = C3 * HW
    dst_offset_3 = (C0 + C1 + C2) * HW
    copy_kernel[((n3 + BLOCK_SIZE - 1) // BLOCK_SIZE,)](
        src_ptr=in_6, dst_ptr=out.view(-1)[dst_offset_3:], n_elements=n3, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # tmp_7 goes to channels [C0+C1+C2+C3, C0+C1+C2+C3+C4)
    n4 = C4 * HW
    dst_offset_4 = (C0 + C1 + C2 + C3) * HW
    copy_kernel[((n4 + BLOCK_SIZE - 1) // BLOCK_SIZE,)](
        src_ptr=tmp_7, dst_ptr=out.view(-1)[dst_offset_4:], n_elements=n4, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


def pattern(in_5, in_7, in_8, in_6, tmp_7):
    result = torch.cat([in_5, in_7, in_8, in_6, tmp_7], dim=1)
    return result


def replacement_args(in_5, in_7, in_8, in_6, tmp_7):
    return (in_5, in_7, in_8, in_6, tmp_7)


def replacement_func():
    return triton_cat