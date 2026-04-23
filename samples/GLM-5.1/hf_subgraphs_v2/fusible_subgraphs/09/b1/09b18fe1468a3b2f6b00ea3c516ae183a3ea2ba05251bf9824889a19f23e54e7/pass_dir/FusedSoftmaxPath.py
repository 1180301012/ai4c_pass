import torch
import triton
import triton.language as tl


def pattern(in_1, in_2, in_3):
    tmp_1 = in_1 - in_2
    tmp_2 = tmp_1.pow(2)
    tmp_3 = tmp_2.sum(dim=3)
    tmp_4 = in_3 * tmp_3
    tmp_5 = torch.nn.functional.softmax(tmp_4, dim=2)
    tmp_9 = tmp_5.unsqueeze(3)
    return tmp_9


def replacement_args(in_1, in_2, in_3):
    return (in_1, in_2, in_3, "route_softmax")


# ==================== Triton Kernel ====================

@triton.jit
def fused_softmax_kernel(
    in_1_ptr, in_2_ptr, in_3_ptr, out_ptr,
    stride_in1_p, stride_in1_k, stride_in1_f,
    stride_in2_k, stride_in2_f,
    stride_in3_k,
    stride_out_p, stride_out_k,
    num_codewords: tl.constexpr, num_features: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    p = tl.program_id(0)
    dists = tl.zeros([num_codewords], dtype=tl.float32)
    k_offsets = tl.arange(0, num_codewords)

    for f_start in range(0, num_features, BLOCK_F):
        f_offsets = f_start + tl.arange(0, BLOCK_F)
        f_mask = f_offsets < num_features

        # Load in_1[p, k, f] for all codewords
        in_1_offsets = p * stride_in1_p + k_offsets[:, None] * stride_in1_k + f_offsets[None, :] * stride_in1_f
        in_1_vals = tl.load(in_1_ptr + in_1_offsets, mask=f_mask[None, :], other=0.0).to(tl.float32)

        # Load in_2[k, f] for all codewords
        in_2_offsets = k_offsets[:, None] * stride_in2_k + f_offsets[None, :] * stride_in2_f
        in_2_vals = tl.load(in_2_ptr + in_2_offsets, mask=f_mask[None, :], other=0.0).to(tl.float32)

        # Compute squared difference and accumulate
        diff = in_1_vals - in_2_vals
        sq_diff = diff * diff
        dists += tl.sum(sq_diff, axis=1)

    # Multiply by scale
    in_3_vals = tl.load(in_3_ptr + k_offsets * stride_in3_k).to(tl.float32)
    scaled = in_3_vals * dists

    # Softmax (numerically stable)
    max_val = tl.max(scaled, axis=0)
    exp_vals = tl.exp(scaled - max_val)
    sum_exp = tl.sum(exp_vals, axis=0)
    softmax_out = exp_vals / sum_exp

    # Store output
    out_offsets = p * stride_out_p + k_offsets * stride_out_k
    tl.store(out_ptr + out_offsets, softmax_out.to(out_ptr.dtype.element_ty))


@torch.fx.wrap
def fused_softmax_path(in_1, in_2, in_3):
    num_pixels = in_1.shape[1]
    num_codewords = in_1.shape[2]
    num_features = in_1.shape[3]

    out = torch.empty((1, num_pixels, num_codewords, 1), dtype=in_1.dtype, device=in_1.device)

    BLOCK_F = 128
    fused_softmax_kernel[(num_pixels,)](
        in_1_ptr=in_1, in_2_ptr=in_2, in_3_ptr=in_3, out_ptr=out,
        stride_in1_p=in_1.stride(1), stride_in1_k=in_1.stride(2), stride_in1_f=in_1.stride(3),
        stride_in2_k=in_2.stride(2), stride_in2_f=in_2.stride(3),
        stride_in3_k=in_3.stride(2),
        stride_out_p=out.stride(1), stride_out_k=out.stride(2),
        num_codewords=num_codewords, num_features=num_features,
        BLOCK_F=BLOCK_F,
    )

    return out


@torch.fx.wrap
def _placeholder_broadcast_sub(*args):
    # Placeholder for shared replacement_func routing - never actually called for this pass
    pass


@torch.fx.wrap
def dispatch_wrapper(*args):
    route = args[-1]
    if route == "route_softmax":
        return fused_softmax_path(args[0], args[1], args[2])
    elif route == "route_broadcast_sub":
        return _placeholder_broadcast_sub(args[0], args[4])
    else:
        raise ValueError(f"Unknown route: {route}")


def replacement_func():
    return dispatch_wrapper