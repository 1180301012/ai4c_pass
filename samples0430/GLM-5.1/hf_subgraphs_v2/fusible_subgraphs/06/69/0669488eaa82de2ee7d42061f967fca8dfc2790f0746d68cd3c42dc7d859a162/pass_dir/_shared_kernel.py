import torch
import triton
import triton.language as tl

SCALE_A = 0.14433756729740643
SCALE_B = 0.07216878364870322


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_K': 128}, num_warps=4),
        triton.Config({'BLOCK_K': 256}, num_warps=4),
        triton.Config({'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_K': 128}, num_warps=8),
        triton.Config({'BLOCK_K': 256}, num_warps=8),
    ],
    key=['K'],
)
@triton.jit
def fused_relu_norm_kernel(
    in_1_ptr, in_0_ptr, out_ptr,
    B, C, K,
    stride_in_b, stride_in_c, stride_in_k,
    stride_out_b, stride_out_c, stride_out_k,
    scale,
    BLOCK_K: tl.constexpr,
):
    bc_idx = tl.program_id(0)
    b_idx = bc_idx // C
    c_idx = bc_idx % C

    # Load the weight scalar
    weight = tl.load(in_0_ptr).to(tl.float32)

    # Phase 1: Compute sum of squares for norm (accumulate in float32)
    sum_sq = 0.0
    base_in = b_idx * stride_in_b + c_idx * stride_in_c
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K
        ptrs = in_1_ptr + base_in + k_offsets * stride_in_k
        values = tl.load(ptrs, mask=k_mask, other=0.0).to(tl.float32)
        relu_values = tl.maximum(values, 0.0)
        sum_sq += tl.sum(relu_values * relu_values)

    # Compute denominator: max(norm * scale, eps)
    norm_val = tl.sqrt(sum_sq)
    denom = tl.maximum(norm_val * scale, 1e-5)

    # Phase 2: Normalize and store
    base_out = b_idx * stride_out_b + c_idx * stride_out_c
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K
        in_ptrs = in_1_ptr + base_in + k_offsets * stride_in_k
        out_ptrs = out_ptr + base_out + k_offsets * stride_out_k
        values = tl.load(in_ptrs, mask=k_mask, other=0.0).to(tl.float32)
        relu_values = tl.maximum(values, 0.0)
        result = relu_values * weight / denom
        tl.store(out_ptrs, result, mask=k_mask)


def _fused_relu_norm_impl(in_0, in_1, scale_val):
    in_1_flat = in_1.flatten(2)
    B, C, K = in_1_flat.shape

    stride_in = in_1_flat.stride()
    stride_in_b = stride_in[0]
    stride_in_c = stride_in[1]
    stride_in_k = stride_in[2]

    out = torch.empty((B, C, K), dtype=in_1.dtype, device=in_1.device)
    stride_out = out.stride()
    stride_out_b = stride_out[0]
    stride_out_c = stride_out[1]
    stride_out_k = stride_out[2]

    grid = (B * C,)

    fused_relu_norm_kernel[grid](
        in_1_ptr=in_1_flat,
        in_0_ptr=in_0,
        out_ptr=out,
        B=B, C=C, K=K,
        stride_in_b=stride_in_b,
        stride_in_c=stride_in_c,
        stride_in_k=stride_in_k,
        stride_out_b=stride_out_b,
        stride_out_c=stride_out_c,
        stride_out_k=stride_out_k,
        scale=scale_val,
    )

    return out


@torch.fx.wrap
def fused_relu_norm_dispatch(in_0, in_1, route):
    if route == "route_scale_a":
        return _fused_relu_norm_impl(in_0, in_1, SCALE_A)
    elif route == "route_scale_b":
        return _fused_relu_norm_impl(in_0, in_1, SCALE_B)
    else:
        raise ValueError(f"Unknown route: {route}")


def replacement_func():
    return fused_relu_norm_dispatch