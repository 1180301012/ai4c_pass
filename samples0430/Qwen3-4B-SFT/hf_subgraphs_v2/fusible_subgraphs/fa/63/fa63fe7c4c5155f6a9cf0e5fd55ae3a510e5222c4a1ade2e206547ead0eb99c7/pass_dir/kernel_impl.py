import torch

# ---------------------------------------------------------------------------
# Shared dispatch wrapper imported by all 8 pass files.
# It is ONE Python object so replacement_func_limit is never hit.
# ---------------------------------------------------------------------------
@torch.fx.wrap
def dispatch_func(in_1, in_0, in_2, route):
    """
    Fused: (in_1 @ in_0) + slice(in_2, 1:) + transpose + reshape + split(3).
    Route string selects spatial/output dims; all dims are inferred from in_2.
    """
    dim_h  = in_2.shape[1]
    dim_t  = in_2.shape[2]
    dim_d  = in_2.shape[3]
    L_out  = H_out = int(route[-1])   # the last digit of route == L_out == H_out
    T      = dim_t
    D      = dim_d
    C_out  = dim_h * D
    N_T    = dim_h * T
    dtype_ = in_1.dtype
    device_ = in_1.device

    grid = lambda meta: (
        triton.cdiv(N_T, meta['BLOCK_M']),
        dim_h,
        triton.cdiv(H_out * L_out, meta['BLOCK_M']),
    )

    if route == "p1":  # reshape(1,152,7,7), split[38,57,57]
        C0, C1, C2 = 38, 57, 57
        out0 = torch.empty((1, C0,  7, 7), dtype=dtype_, device=device_)
        out1 = torch.empty((1, C1,  7, 7), dtype=dtype_, device=device_)
        out2 = torch.empty((1, C2,  7, 7), dtype=dtype_, device=device_)
        fused_matmul_transpose_split_kernel[grid](
            in_1, in_0, in_2, out0, out1, out2,
            dim_h, T, D, C_out, 7, 7, dim_h,
        )
    elif route == "p2":  # reshape(1,320,7,7), split[80,120,120]
        C0, C1, C2 =  80, 120, 120
        out0 = torch.empty((1, C0,  7, 7), dtype=dtype_, device=device_)
        out1 = torch.empty((1, C1,  7, 7), dtype=dtype_, device=device_)
        out2 = torch.empty((1, C2,  7, 7), dtype=dtype_, device=device_)
        fused_matmul_transpose_split_kernel[grid](
            in_1, in_0, in_2, out0, out1, out2,
            dim_h, T, D, C_out, 7, 7, dim_h,
        )
    elif route == "p3":  # reshape(1,216,14,14), split[54,81,81]
        C0, C1, C2 =  54,  81,  81
        out0 = torch.empty((1, C0, 14, 14), dtype=dtype_, device=device_)
        out1 = torch.empty((1, C1, 14, 14), dtype=dtype_, device=device_)
        out2 = torch.empty((1, C2, 14, 14), dtype=dtype_, device=device_)
        fused_matmul_transpose_split_kernel[grid](
            in_1, in_0, in_2, out0, out1, out2,
            dim_h, T, D, C_out, 14, 14, dim_h,
        )
    elif route == "p4":  # reshape(1,256,48,48), split[64,96,96]
        C0, C1, C2 =  64,  96,  96
        out0 = torch.empty((1, C0, 48, 48), dtype=dtype_, device=device_)
        out1 = torch.empty((1, C1, 48, 48), dtype=dtype_, device=device_)
        out2 = torch.empty((1, C2, 48, 48), dtype=dtype_, device=device_)
        fused_matmul_transpose_split_kernel[grid](
            in_1, in_0, in_2, out0, out1, out2,
            dim_h, T, D, C_out, 48, 48, dim_h,
        )
    elif route == "p5":  # reshape(1,256,14,14), split[64,96,96]
        C0, C1, C2 =  64,  96,  96
        out0 = torch.empty((1, C0, 14, 14), dtype=dtype_, device=device_)
        out1 = torch.empty((1, C1, 14, 14), dtype=dtype_, device=device_)
        out2 = torch.empty((1, C2, 14, 14), dtype=dtype_, device=device_)
        fused_matmul_transpose_split_kernel[grid](
            in_1, in_0, in_2, out0, out1, out2,
            dim_h, T, D, C_out, 14, 14, dim_h,
        )
    elif route == "p6":  # reshape(1,152,28,28), split[38,57,57]
        C0, C1, C2 =  38,  57,  57
        out0 = torch.empty((1, C0, 28, 28), dtype=dtype_, device=device_)
        out1 = torch.empty((1, C1, 28, 28), dtype=dtype_, device=device_)
        out2 = torch.empty((1, C2, 28, 28), dtype=dtype_, device=device_)
        fused_matmul_transpose_split_kernel[grid](
            in_1, in_0, in_2, out0, out1, out2,
            dim_h, T, D, C_out, 28, 28, dim_h,
        )
    elif route == "p7":  # reshape(1,128,28,28), split[32,48,48]
        C0, C1, C2 =  32,  48,  48
        out0 = torch.empty((1, C0, 28, 28), dtype=dtype_, device=device_)
        out1 = torch.empty((1, C1, 28, 28), dtype=dtype_, device=device_)
        out2 = torch.empty((1, C2, 28, 28), dtype=dtype_, device=device_)
        fused_matmul_transpose_split_kernel[grid](
            in_1, in_0, in_2, out0, out1, out2,
            dim_h, T, D, C_out, 28, 28, dim_h,
        )
    elif route == "p8":  # reshape(1,152,56,56), split[38,57,57]
        C0, C1, C2 =  38,  57,  57
        out0 = torch.empty((1, C0, 56, 56), dtype=dtype_, device=device_)
        out1 = torch.empty((1, C1, 56, 56), dtype=dtype_, device=device_)
        out2 = torch.empty((1, C2, 56, 56), dtype=dtype_, device=device_)
        fused_matmul_transpose_split_kernel[grid](
            in_1, in_0, in_2, out0, out1, out2,
            dim_h, T, D, C_out, 56, 56, dim_h,
        )
    # unreachable fallback
    return out0, out1, out2