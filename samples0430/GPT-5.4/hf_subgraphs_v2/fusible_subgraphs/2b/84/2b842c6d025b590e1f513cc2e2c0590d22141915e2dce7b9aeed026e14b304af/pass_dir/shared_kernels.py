import torch
import triton
import triton.language as tl


@triton.jit
def masked_sub_kernel(
    mask_ptr,
    x_ptr,
    out_ptr,
    dim0,
    dim1,
    mask_s0,
    mask_s1,
    x_s0,
    x_s1,
    x_s2,
    out_s0,
    out_s1,
    out_s2,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = dim0 * dim1 * 2
    mask = offs < total

    row = offs // 2
    ch = offs % 2
    i0 = row // dim1
    i1 = row % dim1

    mask_offs = i0 * mask_s0 + i1 * mask_s1
    x_offs = i0 * x_s0 + i1 * x_s1 + ch * x_s2
    out_offs = i0 * out_s0 + i1 * out_s1 + ch * out_s2

    mask_vals = tl.load(mask_ptr + mask_offs, mask=mask, other=0)
    x_vals = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)
    scaled_mask = tl.cast(mask_vals, tl.float32) * 1000000.0
    out = x_vals - scaled_mask
    tl.store(out_ptr + out_offs, out, mask=mask)


@triton.jit
def squeeze_lastdim1_kernel(
    in_ptr,
    out_ptr,
    dim0,
    dim1,
    in_s0,
    in_s1,
    out_s0,
    out_s1,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = dim0 * dim1
    mask = offs < total

    i0 = offs // dim1
    i1 = offs % dim1

    in_offs = i0 * in_s0 + i1 * in_s1
    out_offs = i0 * out_s0 + i1 * out_s1
    vals = tl.load(in_ptr + in_offs, mask=mask, other=0.0)
    tl.store(out_ptr + out_offs, vals, mask=mask)


@torch.fx.wrap
def shared_dispatch(*args):
    route = args[-1]

    if route == "masked_sub":
        in_0, in_1, _ = args
        mask = in_0 if in_0.device == in_1.device else torch.as_tensor(in_0, device=in_1.device)

        out = torch.empty(in_1.shape, device=in_1.device, dtype=torch.float32)

        if in_1.is_cuda:
            dim0 = in_1.shape[0]
            dim1 = in_1.shape[1]

            mask_stride = mask.stride()
            x_stride = in_1.stride()
            out_stride = out.stride()

            block = 64
            grid = (triton.cdiv(dim0 * dim1 * 2, block),)
            masked_sub_kernel[grid](
                mask,
                in_1,
                out,
                dim0,
                dim1,
                mask_stride[0],
                mask_stride[1],
                x_stride[0],
                x_stride[1],
                x_stride[2],
                out_stride[0],
                out_stride[1],
                out_stride[2],
                BLOCK_SIZE=block,
                num_warps=1,
                num_stages=1,
            )
            return out

        return in_1 - mask * 1000000.0

    if route == "squeeze_contiguous":
        x, _ = args
        out = torch.empty(x.shape[:-1], device=x.device, dtype=x.dtype)

        if x.is_cuda:
            dim0 = x.shape[0]
            dim1 = x.shape[1]
            in_stride = x.stride()
            out_stride = out.stride()
            block = 64
            grid = (triton.cdiv(dim0 * dim1, block),)
            squeeze_lastdim1_kernel[grid](
                x,
                out,
                dim0,
                dim1,
                in_stride[0],
                in_stride[1],
                out_stride[0],
                out_stride[1],
                BLOCK_SIZE=block,
                num_warps=1,
                num_stages=1,
            )
            return out

        return x.squeeze(-1).contiguous()

    raise RuntimeError(f"Unknown route: {route}")