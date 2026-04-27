import torch
import triton
import triton.language as tl


def pattern(in_3, in_4, tmp_3):
    tmp_4 = torch.cat([in_3, in_4, tmp_3], 2)
    tmp_5 = tmp_4.sigmoid()
    tmp_6 = tmp_5 - 0.25
    tmp_7 = tmp_6 * 3.141592653589793
    return tmp_7


def replacement_args(in_3, in_4, tmp_3):
    return (in_3, in_4, tmp_3)


@triton.jit
def _fused_seg_kernel(
    src_ptr, dst_ptr,
    S_src, S_total, dst_offset,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Processes one source segment for the fused cat+sigmoid+sub+mul operation.
    2D grid: (ceil(S_src/BLOCK_SIZE), B)
      - tl.program_id(0): block index within the segment
      - tl.program_id(1): batch index
    Reads are fully coalesced; no divmod needed.
    """
    blk = tl.program_id(0)
    b   = tl.program_id(1)

    pos  = blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = pos < S_src

    # Coalesced load from source
    val = tl.load(src_ptr + b * S_src + pos, mask=mask, other=0.0).to(tl.float32)

    # Fused ops: sigmoid, subtract 0.25, multiply by pi
    val = tl.sigmoid(val)
    val = (val - 0.25) * 3.141592653589793

    # Write to the correct slice in the output
    tl.store(dst_ptr + b * S_total + dst_offset + pos, val, mask=mask)


@torch.fx.wrap
def fused_cat_sigmoid_sub_mul(in_3, in_4, tmp_3):
    B      = in_3.shape[0]
    S1     = in_3.shape[2]   # typically 6400
    S2     = in_4.shape[2]   # typically 1600
    S3     = tmp_3.shape[2]  # typically  400
    S_total = S1 + S2 + S3   # typically 8400

    out = torch.empty((B, 1, S_total), dtype=in_3.dtype, device=in_3.device)

    BLOCK_SIZE = 512

    # Launch one kernel per source segment using a 2D grid (blocks, batch).
    # This avoids expensive integer divmod by non-power-of-2 strides and keeps reads coalesced.
    NB1 = triton.cdiv(S1, BLOCK_SIZE)
    NB2 = triton.cdiv(S2, BLOCK_SIZE)
    NB3 = triton.cdiv(S3, BLOCK_SIZE)

    _fused_seg_kernel[(NB1, B)](in_3,  out, S1, S_total, 0,        BLOCK_SIZE=BLOCK_SIZE)
    _fused_seg_kernel[(NB2, B)](in_4,  out, S2, S_total, S1,       BLOCK_SIZE=BLOCK_SIZE)
    _fused_seg_kernel[(NB3, B)](tmp_3, out, S3, S_total, S1 + S2,  BLOCK_SIZE=BLOCK_SIZE)

    return out


def replacement_func():
    return fused_cat_sigmoid_sub_mul