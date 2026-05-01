import torch
import triton
import triton.language as tl


@triton.jit
def batched_matvec_kernel(
    in1_ptr,   # [B, M, K]  left matrix (contiguous)
    in0_ptr,   # [B, K, 1]  right vector (contiguous; in0[b,k,0] = in0_ptr + b*K + k)
    out_ptr,   # [B*M]      output (flattened, contiguous)
    B, M, K,
    BLOCK_K: tl.constexpr,
):
    """
    Computes out[b*M + m] = sum_k( in1[b, m, k] * in0[b, k, 0] )
    Each program handles one (b, m) output element.
    K is always 9; BLOCK_K=16 covers it in one shot with masking.
    """
    pid = tl.program_id(0)
    b = pid // M
    m = pid % M

    k_offsets = tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K

    a = tl.load(in1_ptr + b * M * K + m * K + k_offsets,
                mask=k_mask, other=0.0).to(tl.float32)
    bv = tl.load(in0_ptr + b * K + k_offsets,
                 mask=k_mask, other=0.0).to(tl.float32)

    result = tl.sum(a * bv, axis=0)
    tl.store(out_ptr + b * M + m, result)


@triton.jit
def transpose_last2_kernel(
    in_ptr,    # [1, H, S, D]  input  (in[0,h,s,d] = in_ptr + h*S*D + s*D + d)
    out_ptr,   # [1, H, D, S]  output (out[0,h,d,s] = out_ptr + h*D*S + d*S + s)
    H, S, D,
    BLOCK_SIZE: tl.constexpr,
):
    """Transposes the last two dims: [1,H,S,D] -> [1,H,D,S]."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = H * S * D
    mask = offsets < total

    h = offsets // (D * S)
    rem = offsets % (D * S)
    d = rem // S
    s = rem % S

    in_offsets = h * S * D + s * D + d
    data = tl.load(in_ptr + in_offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, data, mask=mask)


@torch.fx.wrap
def triton_matmul_reshape(in_0, in_1, route):
    """
    Replacement for: matmul(in_1, in_0) -> reshape([-1, N])
      in_0 : [B, K, 1]   right vector
      in_1 : [B, M, K]   left  matrix
      route: str(N)       reshape column count
    """
    N = int(route)
    B = in_1.shape[0]
    M = in_1.shape[1]
    K = in_1.shape[2]
    total_bm = B * M
    out = torch.empty((total_bm // N, N), dtype=in_1.dtype, device=in_1.device)
    batched_matvec_kernel[(total_bm,)](
        in_1, in_0, out,
        B, M, K,
        BLOCK_K=16,
    )
    return out


@torch.fx.wrap
def triton_transpose_last2(x):
    """
    Replacement for: x.transpose(-1, -2)
      x : [1, H, S, D]  ->  out : [1, H, D, S]
    """
    H = x.shape[1]
    S = x.shape[2]
    D = x.shape[3]
    out = torch.empty((1, H, D, S), dtype=x.dtype, device=x.device)
    total = H * S * D
    BLOCK_SIZE = 256
    grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    transpose_last2_kernel[grid](
        x, out,
        H, S, D,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out