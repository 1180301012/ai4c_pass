import torch
import triton
import triton.language as tl


def pattern(in_1, in_2, in_3):
    """
    Match: sub -> pow(2) -> sum(dim=3) -> scale
    Returns scaled squared-distances [1, N, K] (= tmp_4 in the model).
    Softmax and unsqueeze remain in the original graph.
    """
    tmp_1 = in_1 - in_2
    tmp_2 = tmp_1.pow(2)
    tmp_3 = tmp_2.sum(dim=3)
    tmp_4 = in_3 * tmp_3
    return tmp_4


def replacement_args(in_1, in_2, in_3):
    return (in_1, in_2, in_3)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 64},  num_warps=4),
        triton.Config({'BLOCK_D': 64},  num_warps=8),
        triton.Config({'BLOCK_D': 128}, num_warps=4),
        triton.Config({'BLOCK_D': 128}, num_warps=8),
        triton.Config({'BLOCK_D': 256}, num_warps=4),
        triton.Config({'BLOCK_D': 256}, num_warps=8),
        triton.Config({'BLOCK_D': 512}, num_warps=4),
        triton.Config({'BLOCK_D': 512}, num_warps=8),
        triton.Config({'BLOCK_D': 512}, num_warps=16),
    ],
    key=['N', 'K', 'D'],
)
@triton.jit
def fused_sq_dist_scale_kernel(
    in1_ptr, in2_ptr, in3_ptr, out_ptr,
    N, K, D,
    BLOCK_K: tl.constexpr,    # = 32
    BLOCK_D: tl.constexpr,    # tile size along D
):
    """
    Grid = (N,). Each program handles one n value, all K=32 codewords.
    Fuses: (in1[n,:,:] - in2[0,:,:])^2 -> sum over D -> scale by in3
    Computes diff and squared-diff in native dtype (fp16/bf16) to match
    the original model's fp16/bf16 precision exactly.
    """
    n = tl.program_id(0)
    k_range = tl.arange(0, BLOCK_K)  # [K]

    # Accumulate in fp32 to avoid overflow (matches PyTorch's internal accumulation)
    dist = tl.zeros([BLOCK_K], dtype=tl.float32)

    # Tile over D in chunks of BLOCK_D
    for chunk in range(D // BLOCK_D):
        d_range = chunk * BLOCK_D + tl.arange(0, BLOCK_D)

        # in1[0, n, k, d] -> n*K*D + k*D + d
        in1_offs = n * BLOCK_K * D + k_range[:, None] * D + d_range[None, :]
        in1_data = tl.load(in1_ptr + in1_offs, eviction_policy='evict_first')   # [K, BLOCK_D] native dtype

        # in2[0, 0, k, d] -> k*D + d
        in2_offs = k_range[:, None] * D + d_range[None, :]
        in2_data = tl.load(in2_ptr + in2_offs, eviction_policy='evict_last')   # [K, BLOCK_D] native dtype

        # Compute diff and squared-diff in native dtype (fp16/bf16)
        # This exactly matches: tmp_1=in1-in2 (fp16), tmp_2=tmp_1.pow(2) (fp16)
        diff = in1_data - in2_data                # [K, BLOCK_D] native
        sq   = diff * diff                         # [K, BLOCK_D] native

        # Sum in fp32 to match PyTorch's internal fp32 accumulation for sum(dim=3)
        dist += tl.sum(sq.to(tl.float32), axis=1)  # [K] fp32

    # Convert accumulated fp32 sum -> native dtype (= tmp_3 in fp16/bf16)
    elem_ty = in1_ptr.dtype.element_ty
    dist_native = dist.to(elem_ty)                # [K] native

    # Scale: tmp_4 = in3 * tmp_3, both in native dtype
    in3 = tl.load(in3_ptr + k_range)             # [K] native
    scaled = in3 * dist_native                    # [K] native

    # Store: out[0, n, k] -> n*K + k
    tl.store(out_ptr + n * BLOCK_K + k_range, scaled)


@torch.fx.wrap
def fused_sq_dist_scale(in_1, in_2, in_3):
    """
    in_1: [1, N, K, D]  e.g. [1, 4096, 32, 512]
    in_2: [1, 1, K, D]  e.g. [1, 1,   32, 512]
    in_3: [1, 1, K]     e.g. [1, 1,   32]
    Returns: [1, N, K]  scaled squared distances (= tmp_4 in the model)
    """
    N = in_1.shape[1]  # 4096
    K = in_1.shape[2]  # 32
    D = in_1.shape[3]  # 512

    out = torch.empty((1, N, K), dtype=in_1.dtype, device=in_1.device)

    in_1_c = in_1.contiguous()
    in_2_c = in_2.contiguous()
    in_3_c = in_3.contiguous()

    fused_sq_dist_scale_kernel[(N,)](
        in_1_c, in_2_c, in_3_c, out,
        N, K, D,
        BLOCK_K=K,
    )
    return out


def replacement_func():
    return fused_sq_dist_scale