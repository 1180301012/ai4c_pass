"""
Pass for: in_1 @ in_0 + in_1[:, :, 1:, :] + in_2[:,:,1:,:].T.reshape(1,152,7,7).split([38,57,57],1)
Matches: coat_tiny (float32/float16/bf16), reshape(1,152,7,7), split [38,57,57]
"""
import torch
from pass_dir.kernel_impl import dispatch_func


# ---------------------------------------------------------------------------
# Pattern
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2):
    tmp_0 = in_1 @ in_0
    tmp_1 = in_1[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
    tmp_2 = in_2[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
    tmp_3 = tmp_2.transpose(-1, -2)
    tmp_4 = tmp_3.reshape(1, 152, 7, 7)
    tmp_5 = torch.functional.split(tmp_4, [38, 57, 57], dim=1)
    tmp_6 = tmp_5[0]
    tmp_7 = tmp_5[1]
    tmp_8 = tmp_5[2]
    return (tmp_0, tmp_6, tmp_7, tmp_8, tmp_1)


# ---------------------------------------------------------------------------
# Replacement args
# ---------------------------------------------------------------------------
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "p1")


# ---------------------------------------------------------------------------
# Shared dispatch wrapper (identical across all pass files)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _dispatch_matmul_transpose_split(in_1, in_0, in_2, route):
    # Shapes are derived at runtime from in_2 (contiguous [1,H,T,D])
    dim_h    = in_2.shape[1]             # H  (number of heads, same for all inputs)
    dim_t    = in_2.shape[2]             # T  (token count, == dim_t in kernel)
    dim_d    = in_2.shape[3]             # D  (head dim, in kernel as dim_d)
    T        = dim_t                    # after CLS, still dim_t tokens
    D        = dim_d
    N_T_out  = dim_h * T                # total output tokens == N_T
    C_out    = dim_h * D                # total output channels
    H_out    = 7                        # output height (fixed for 7×7 spatial)
    L_out    = 7                        # output width
    dtype_   = in_1.dtype
    device   = in_1.device

    out0 = torch.empty((1, 38,  7, 7), dtype=dtype_, device=device)
    out1 = torch.empty((1, 57,  7, 7), dtype=dtype_, device=device)
    out2 = torch.empty((1, 57,  7, 7), dtype=dtype_, device=device)

    # Grid: (token_tiles, 1, yx_tiles)
    #        dim 0: N_T_out tiles  (BLOCK_M from autotune)
    #        dim 2: used for yx indexing in the store stage
    #        We allocate a no-op "out2_new" for pid_yx_rem=1..L_out-1
    out2_new = torch.empty((1, 57,  7, 7), dtype=dtype_, device=device)

    grid = lambda meta: (
        triton.cdiv(N_T_out, meta['BLOCK_M']),
        dim_h,
        triton.cdiv(7 * 7, meta['BLOCK_M']),
    )

    fused_matmul_transpose_split_kernel[grid](
        in_1, in_0, in_2,
        out0, out1, out2,
        dim_h, T, D,
        C_out, H_out, L_out, dim_h,
    )
    return out0, out1, out2


# ---------------------------------------------------------------------------
# replacement_func
# ---------------------------------------------------------------------------
def replacement_func():
    return dispatch_func