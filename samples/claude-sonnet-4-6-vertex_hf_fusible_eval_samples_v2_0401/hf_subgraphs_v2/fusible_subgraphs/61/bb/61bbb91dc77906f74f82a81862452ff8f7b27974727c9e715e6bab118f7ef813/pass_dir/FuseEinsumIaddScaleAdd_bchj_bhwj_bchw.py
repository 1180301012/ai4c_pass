import operator
import torch
import torch.fx
import triton
import triton.language as tl


# Patch torch.fx.Proxy so that `a += b` in the pattern creates
# a call_function(operator.iadd, ...) node instead of falling back to add.
def _proxy_iadd(self, other):
    return self.tracer.create_proxy('call_function', operator.iadd, (self, other), {})

torch.fx.Proxy.__iadd__ = _proxy_iadd


def pattern(in_0, in_1, in_2, in_3, in_4):
    einsum = torch.functional.einsum('bchj,bhwj->bchw', in_4, in_1)
    in_3 += einsum          # now correctly traces as operator.iadd
    tmp_3 = in_3 * in_0
    tmp_4 = tmp_3 + in_2
    tmp_5 = tmp_4.contiguous()
    return tmp_5


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    ],
    key=['B', 'C', 'H', 'W', 'J'],
)
@triton.jit
def fused_einsum_scale_add_kernel(
    # in_4: [B, C, H, J]
    in4_ptr, s4_b, s4_c, s4_h, s4_j,
    # in_1: [B, H, W, J]
    in1_ptr, s1_b, s1_h, s1_w, s1_j,
    # in_3: [B, C, H, W]
    in3_ptr, s3_b, s3_c, s3_h, s3_w,
    # in_2: [B, C, H, W]
    in2_ptr, s2_b, s2_c, s2_h, s2_w,
    # scale (scalar)
    scale_ptr,
    # output: [B, C, H, W]
    out_ptr, so_b, so_c, so_h, so_w,
    # dimensions
    B, C, H, W, J,
    # tile sizes (constexpr)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Grid dims: axis-0 = B*H, axis-1 = C blocks, axis-2 = W blocks
    bh    = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    b = bh // H
    h = bh % H

    # Row / col ranges for this tile
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_m = rm < C
    mask_n = rn < W

    # Pointers to the (b, h) slice
    p_a  = in4_ptr + b * s4_b + h * s4_h   # A[c, j] = in_4[b, c, h, j]
    p_bt = in1_ptr + b * s1_b + h * s1_h   # B[j, w] = in_1[b, h, w, j]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, J, BLOCK_K):
        rk     = k + tl.arange(0, BLOCK_K)   # [BLOCK_K]
        mask_k = rk < J

        # A [BLOCK_M, BLOCK_K]: in_4[b, rm, h, rk]
        a_offs = rm[:, None] * s4_c + rk[None, :] * s4_j
        a_mask = mask_m[:, None] & mask_k[None, :]
        a = tl.load(p_a + a_offs, mask=a_mask, other=0.0)

        # B^T [BLOCK_K, BLOCK_N]: in_1[b, h, rn, rk] viewed as [rk, rn]
        bt_offs = rk[:, None] * s1_j + rn[None, :] * s1_w
        bt_mask = mask_k[:, None] & mask_n[None, :]
        bt = tl.load(p_bt + bt_offs, mask=bt_mask, other=0.0)

        # acc += A @ B^T  (float32 accumulation)
        acc += tl.dot(a.to(tl.float32), bt.to(tl.float32))

    # ---- Epilogue: (in_3 + acc) * scale + in_2 ----
    p_in3 = in3_ptr + b * s3_b + h * s3_h
    p_in2 = in2_ptr + b * s2_b + h * s2_h
    p_out = out_ptr + b * so_b + h * so_h

    ew_mask = mask_m[:, None] & mask_n[None, :]
    ew_in3  = rm[:, None] * s3_c + rn[None, :] * s3_w
    ew_in2  = rm[:, None] * s2_c + rn[None, :] * s2_w
    ew_out  = rm[:, None] * so_c + rn[None, :] * so_w

    in3_val = tl.load(p_in3 + ew_in3, mask=ew_mask, other=0.0)
    in2_val = tl.load(p_in2 + ew_in2, mask=ew_mask, other=0.0)
    scale   = tl.load(scale_ptr).to(tl.float32)

    result_f32 = (in3_val.to(tl.float32) + acc) * scale + in2_val.to(tl.float32)

    # Convert back to the original element type and store
    tl.store(p_out + ew_out, result_f32.to(in3_val.dtype), mask=ew_mask)


@torch.fx.wrap
def fused_einsum_iadd_scale_add(in_0, in_1, in_2, in_3, in_4):
    B, C, H, J = in_4.shape
    W = in_1.shape[2]

    out = torch.empty_like(in_3)

    grid = lambda meta: (
        B * H,
        triton.cdiv(C, meta['BLOCK_M']),
        triton.cdiv(W, meta['BLOCK_N']),
    )

    fused_einsum_scale_add_kernel[grid](
        in_4, in_4.stride(0), in_4.stride(1), in_4.stride(2), in_4.stride(3),
        in_1, in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
        in_3, in_3.stride(0), in_3.stride(1), in_3.stride(2), in_3.stride(3),
        in_2, in_2.stride(0), in_2.stride(1), in_2.stride(2), in_2.stride(3),
        in_0,
        out,  out.stride(0),  out.stride(1),  out.stride(2),  out.stride(3),
        B, C, H, W, J,
    )

    return out


def replacement_func():
    return fused_einsum_iadd_scale_add