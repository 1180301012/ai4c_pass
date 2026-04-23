import torch
import triton
import triton.language as tl
import math


@triton.jit
def fused_add_layernorm_kernel(
    # Pointers
    in_2_ptr, rolled_ptr, weight_ptr, bias_ptr,
    out_add_ptr, out_norm_ptr,
    # Strides for in_2 (3D: [1, HW, C])
    in_2_stride_0, in_2_stride_1, in_2_stride_2,
    # Strides for rolled (3D: [1, HW, C])  
    rolled_stride_0, rolled_stride_1, rolled_stride_2,
    # Strides for output (3D: [1, HW, C])
    out_stride_0, out_stride_1, out_stride_2,
    # Dimensions
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    if row_id >= n_rows:
        return

    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < n_cols

    # Load from in_2 at [0, row_id, :]
    in_2_offset = in_2_stride_1 * row_id + in_2_stride_2 * col_offsets
    in_2_val = tl.load(in_2_ptr + in_2_offset, mask=col_mask, other=0.0).to(tl.float32)

    # Load rolled/sliced value
    rolled_offset = rolled_stride_1 * row_id + rolled_stride_2 * col_offsets
    rolled_val = tl.load(rolled_ptr + rolled_offset, mask=col_mask, other=0.0).to(tl.float32)

    # Add: result = in_2 + rolled
    add_val = in_2_val + rolled_val

    # Store add result (convert back to original dtype)
    out_add_offset = out_stride_1 * row_id + out_stride_2 * col_offsets
    tl.store(out_add_ptr + out_add_offset, add_val, mask=col_mask)

    # Layer norm: compute mean and variance across C dimension
    mean = tl.sum(add_val, axis=0) / n_cols
    diff = add_val - mean
    var = tl.sum(diff * diff, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + 1e-05)

    # Load weight and bias
    weight_val = tl.load(weight_ptr + col_offsets, mask=col_mask, other=1.0).to(tl.float32)
    bias_val = tl.load(bias_ptr + col_offsets, mask=col_mask, other=0.0).to(tl.float32)

    # Compute norm output
    norm_val = diff * rstd * weight_val + bias_val

    # Store norm result (convert back to original dtype)
    tl.store(out_norm_ptr + out_stride_1 * row_id + out_stride_2 * col_offsets, norm_val, mask=col_mask)


@torch.fx.wrap
def fused_add_layernorm(in_0, in_1, in_2, in_3):
    import math
    
    # in_3 shape: [1, wH, 7, wW, 7, C] (6D tensor)
    wH = in_3.shape[1]
    ws = in_3.shape[2]
    wW = in_3.shape[3]
    C = in_3.shape[5]
    H_4d = wH * ws
    W_4d = wW * ws

    # Step 1: contiguous + view to 4D
    tmp_2 = in_3.contiguous()
    tmp_3 = tmp_2.view(-1, H_4d, W_4d, C)

    # Step 2: roll by (3, 3) on dims (1, 2)
    tmp_4 = torch.roll(tmp_3, shifts=(3, 3), dims=(1, 2))

    # Step 3: slice [:1, :H_slice, :W_slice, :]
    HW = in_2.shape[1]
    H_slice = int(math.isqrt(HW))
    W_slice = H_slice
    tmp_5 = tmp_4[:1, :H_slice, :W_slice, :]

    # Step 4: contiguous + view to [1, HW, C]
    tmp_6 = tmp_5.contiguous()
    rolled_val = tmp_6.view(1, HW, C)

    n_rows = HW
    n_cols = C

    out_add = torch.empty_like(in_2)
    out_norm = torch.empty_like(in_2)

    in_2_strides = in_2.stride()
    rolled_strides = rolled_val.stride()
    out_strides = out_add.stride()

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    grid = (n_rows,)

    fused_add_layernorm_kernel[grid](
        in_2_ptr=in_2,
        rolled_ptr=rolled_val,
        weight_ptr=in_1,
        bias_ptr=in_0,
        out_add_ptr=out_add,
        out_norm_ptr=out_norm,
        in_2_stride_0=in_2_strides[0],
        in_2_stride_1=in_2_strides[1],
        in_2_stride_2=in_2_strides[2],
        rolled_stride_0=rolled_strides[0],
        rolled_stride_1=rolled_strides[1],
        rolled_stride_2=rolled_strides[2],
        out_stride_0=out_strides[0],
        out_stride_1=out_strides[1],
        out_stride_2=out_strides[2],
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return (out_add, out_norm)


# Pattern matching function - matches the bfloat16/7 variant (133x133, 96)
def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_3.contiguous()
    tmp_3 = tmp_2.view(-1, 133, 133, 96)
    tmp_4 = torch.roll(tmp_3, shifts=(3, 3), dims=(1, 2))
    tmp_5 = tmp_4[slice(None, None, None), slice(None, 128, None), slice(None, 128, None), slice(None, None, None)]
    tmp_6 = tmp_5.contiguous()
    tmp_7 = tmp_6.view(1, 16384, 96)
    tmp_8 = in_2 + tmp_7
    tmp_9 = torch.nn.functional.layer_norm(tmp_8, (96,), in_1, in_0, 1e-05)
    return (tmp_8, tmp_9)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

def replacement_func():
    return fused_add_layernorm