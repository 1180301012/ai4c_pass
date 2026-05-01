import torch
import triton
import triton.language as tl


def pattern(in_3, in_1, in_0, in_4):
    """
    Matches: F.linear + two parallel view+transpose+contiguous chains.
    Replaces 3 CUDA launches (cuBLAS GEMV + 2 contiguous copies) with 1 Triton kernel.
    """
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = in_4.view(1, 1, -1, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    tmp_5 = linear.view(1, 1, -1, 64)
    tmp_6 = tmp_5.transpose(1, 2)
    tmp_9 = tmp_4.contiguous()
    tmp_10 = tmp_6.contiguous()
    return tmp_9, tmp_10


def replacement_args(in_3, in_1, in_0, in_4):
    return (in_3, in_1, in_0, in_4)


@triton.jit
def fused_gemv_copy_kernel(
    x_ptr,    # in_3 flat [512]
    w_ptr,    # in_1 [512, 512] row-major
    b_ptr,    # in_0 [512]
    in4_ptr,  # in_4 flat [512]
    out1_ptr, # tmp_9: in_4 copy -> [1,8,1,64] contiguous (flat [512])
    out2_ptr, # tmp_10: linear output -> [1,8,1,64] contiguous (flat [512])
    BLOCK_M: tl.constexpr,
):
    """
    Fused GEMV + dual copy kernel.
    Each program handles BLOCK_M rows of the 512-row output.
    - Computes GEMV rows for the linear output (out2)
    - Copies in_4 rows to out1
    Memory layout: [1,8,1,64] contiguous == flat [512], so flat index i maps to flat position i.
    """
    pid = tl.program_id(0)
    row_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = tl.arange(0, 512)

    # ---- GEMV part ----
    # Load input x (512 elements, shared across all BLOCK_M rows)
    x_raw = tl.load(x_ptr + col_offsets)
    x_f32 = x_raw.to(tl.float32)

    # Load weight rows [BLOCK_M, 512]
    w = tl.load(w_ptr + row_offsets[:, None] * 512 + col_offsets[None, :]).to(tl.float32)

    # Dot product: [BLOCK_M]
    acc = tl.sum(w * x_f32[None, :], axis=1)

    # Add bias
    b = tl.load(b_ptr + row_offsets).to(tl.float32)
    linear_result = (acc + b).to(x_raw.dtype)

    # ---- Copy part ----
    in4_data = tl.load(in4_ptr + row_offsets)

    # ---- Store both outputs ----
    tl.store(out1_ptr + row_offsets, in4_data)      # tmp_9 = in_4 copy
    tl.store(out2_ptr + row_offsets, linear_result)  # tmp_10 = linear result


@torch.fx.wrap
def fused_linear_dual_reshape(in_3, in_1, in_0, in_4):
    """
    Fused: F.linear(in_3, in_1, in_0) + view+transpose+contiguous for both outputs.
    Inputs:
      in_3: [1,1,512] hidden states
      in_1: [512,512] weight
      in_0: [512] bias
      in_4: [1,1,512] key states
    Outputs:
      out1: [1,8,1,64] = in_4 reshaped (tmp_9)
      out2: [1,8,1,64] = linear output reshaped (tmp_10)
    Replaces 3 CUDA kernel launches with 1.
    """
    out1 = torch.empty((1, 8, 1, 64), dtype=in_4.dtype, device=in_4.device)
    out2 = torch.empty((1, 8, 1, 64), dtype=in_3.dtype, device=in_3.device)

    # Fixed grid: 512 rows / 64 per program = 8 programs
    fused_gemv_copy_kernel[(8,)](
        in_3,  # x: [1,1,512] contiguous -> flat offset i == element i
        in_1,  # w: [512,512] row-major
        in_0,  # b: [512]
        in_4,  # in4: [1,1,512] contiguous -> flat offset i == element i
        out1,  # out1: [1,8,1,64] contiguous -> flat offset i == element i
        out2,  # out2: [1,8,1,64] contiguous -> flat offset i == element i
        BLOCK_M=64,
        num_warps=4,
    )

    return out1, out2


def replacement_func():
    return fused_linear_dual_reshape