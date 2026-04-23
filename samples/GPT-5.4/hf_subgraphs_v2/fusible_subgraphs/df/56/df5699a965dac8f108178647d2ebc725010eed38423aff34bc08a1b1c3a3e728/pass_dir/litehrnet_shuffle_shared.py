import torch
import triton
import triton.language as tl
import graph_net_bench.torch.backend.pass_mgr_backend as pass_mgr_backend


# The default pass-manager dispatch wrapper traces replacements as a single opaque
# leaf call, which collapses multi-output replacements into one value. Disable it
# for these passes so FX can see the 4 explicit outputs from litehrnet_fused().
def _direct_replacement_core_decorator(override_dispatch):
    return pass_mgr_backend.g_replacement_func


pass_mgr_backend.replacement_core_decorator = _direct_replacement_core_decorator


def pattern_impl(batch, in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    conv2d = torch.conv2d(in_6, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.sigmoid(conv2d)
    tmp_4 = in_5 * tmp_3
    tmp_5 = torch.cat([in_2, in_4], dim=1)
    tmp_6 = torch.cat([in_3, tmp_4], dim=1)
    tmp_7 = tmp_5.view(batch, 2, 20, 64, 48)
    tmp_8 = torch.transpose(tmp_7, 1, 2)
    tmp_9 = tmp_8.contiguous()
    tmp_10 = tmp_9.view(batch, 40, 64, 48)
    tmp_11 = tmp_6.view(batch, 2, 40, 32, 24)
    tmp_12 = torch.transpose(tmp_11, 1, 2)
    tmp_13 = tmp_12.contiguous()
    tmp_14 = tmp_13.view(batch, 80, 32, 24)
    chunk = tmp_10.chunk(2, dim=1)
    tmp_16 = chunk[0]
    tmp_17 = chunk[1]
    chunk_1 = tmp_14.chunk(2, dim=1)
    tmp_19 = chunk_1[0]
    tmp_20 = chunk_1[1]
    return (tmp_16, tmp_19, tmp_17, tmp_20)



def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6)


@triton.jit
def shuffle_split20_kernel(
    a_ptr,
    b_ptr,
    out0_ptr,
    out1_ptr,
    HW,
    BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_pair = tl.program_id(1)
    pid_tile = tl.program_id(2)

    offs = pid_tile * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = offs < HW

    src_lo = pid_pair
    src_hi = pid_pair + 10
    out_c_even = pid_pair * 2
    out_c_odd = out_c_even + 1

    base_a_lo = (pid_n * 20 + src_lo) * HW + offs
    base_a_hi = (pid_n * 20 + src_hi) * HW + offs
    base_b_lo = (pid_n * 20 + src_lo) * HW + offs
    base_b_hi = (pid_n * 20 + src_hi) * HW + offs

    a_lo = tl.load(a_ptr + base_a_lo, mask=mask, other=0.0)
    a_hi = tl.load(a_ptr + base_a_hi, mask=mask, other=0.0)
    b_lo = tl.load(b_ptr + base_b_lo, mask=mask, other=0.0)
    b_hi = tl.load(b_ptr + base_b_hi, mask=mask, other=0.0)

    base_out0_even = (pid_n * 20 + out_c_even) * HW + offs
    base_out0_odd = (pid_n * 20 + out_c_odd) * HW + offs
    base_out1_even = (pid_n * 20 + out_c_even) * HW + offs
    base_out1_odd = (pid_n * 20 + out_c_odd) * HW + offs

    tl.store(out0_ptr + base_out0_even, a_lo, mask=mask)
    tl.store(out0_ptr + base_out0_odd, b_lo, mask=mask)
    tl.store(out1_ptr + base_out1_even, a_hi, mask=mask)
    tl.store(out1_ptr + base_out1_odd, b_hi, mask=mask)


@triton.jit
def gated_shuffle_split40_kernel(
    c_ptr,
    d_ptr,
    bias_ptr,
    weight_ptr,
    x_ptr,
    out0_ptr,
    out1_ptr,
    HW,
    BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_pair = tl.program_id(1)
    pid_tile = tl.program_id(2)

    offs = pid_tile * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = offs < HW

    src_lo = pid_pair
    src_hi = pid_pair + 20
    out_c_even = pid_pair * 2
    out_c_odd = out_c_even + 1

    k_offs = tl.arange(0, 16)
    k_mask = k_offs < 10
    x_vals = tl.load(x_ptr + pid_n * 10 + k_offs, mask=k_mask, other=0.0).to(tl.float32)

    w_lo = tl.load(weight_ptr + src_lo * 10 + k_offs, mask=k_mask, other=0.0).to(tl.float32)
    w_hi = tl.load(weight_ptr + src_hi * 10 + k_offs, mask=k_mask, other=0.0).to(tl.float32)
    b_lo = tl.load(bias_ptr + src_lo).to(tl.float32)
    b_hi = tl.load(bias_ptr + src_hi).to(tl.float32)

    acc_lo = tl.sum(x_vals * w_lo, axis=0) + b_lo
    acc_hi = tl.sum(x_vals * w_hi, axis=0) + b_hi
    gate_lo = 1.0 / (1.0 + tl.exp(-acc_lo))
    gate_hi = 1.0 / (1.0 + tl.exp(-acc_hi))

    base_c_lo = (pid_n * 40 + src_lo) * HW + offs
    base_c_hi = (pid_n * 40 + src_hi) * HW + offs
    base_d_lo = (pid_n * 40 + src_lo) * HW + offs
    base_d_hi = (pid_n * 40 + src_hi) * HW + offs

    c_lo = tl.load(c_ptr + base_c_lo, mask=mask, other=0.0)
    c_hi = tl.load(c_ptr + base_c_hi, mask=mask, other=0.0)
    d_lo = tl.load(d_ptr + base_d_lo, mask=mask, other=0.0)
    d_hi = tl.load(d_ptr + base_d_hi, mask=mask, other=0.0)

    out_d_lo = d_lo.to(tl.float32) * gate_lo
    out_d_hi = d_hi.to(tl.float32) * gate_hi

    base_out0_even = (pid_n * 40 + out_c_even) * HW + offs
    base_out0_odd = (pid_n * 40 + out_c_odd) * HW + offs
    base_out1_even = (pid_n * 40 + out_c_even) * HW + offs
    base_out1_odd = (pid_n * 40 + out_c_odd) * HW + offs

    tl.store(out0_ptr + base_out0_even, c_lo, mask=mask)
    tl.store(out0_ptr + base_out0_odd, out_d_lo, mask=mask)
    tl.store(out1_ptr + base_out1_even, c_hi, mask=mask)
    tl.store(out1_ptr + base_out1_odd, out_d_hi, mask=mask)


@torch.fx.wrap
def litehrnet_fused_impl(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    n = in_2.shape[0]
    h1 = in_2.shape[2]
    w1 = in_2.shape[3]
    h2 = in_3.shape[2]
    w2 = in_3.shape[3]

    hw1 = h1 * w1
    hw2 = h2 * w2

    out16 = torch.empty((n, 20, h1, w1), device=in_2.device, dtype=in_2.dtype)
    out17 = torch.empty((n, 20, h1, w1), device=in_2.device, dtype=in_2.dtype)
    out19 = torch.empty((n, 40, h2, w2), device=in_3.device, dtype=in_3.dtype)
    out20 = torch.empty((n, 40, h2, w2), device=in_3.device, dtype=in_3.dtype)

    block_hw1 = 256
    block_hw2 = 256

    grid1 = (n, 10, triton.cdiv(hw1, block_hw1))
    shuffle_split20_kernel[grid1](
        in_2,
        in_4,
        out16,
        out17,
        hw1,
        BLOCK_HW=block_hw1,
        num_warps=4,
        num_stages=2,
    )

    grid2 = (n, 20, triton.cdiv(hw2, block_hw2))
    gated_shuffle_split40_kernel[grid2](
        in_3,
        in_5,
        in_0,
        in_1,
        in_6,
        out19,
        out20,
        hw2,
        BLOCK_HW=block_hw2,
        num_warps=4,
        num_stages=2,
    )

    return (out16, out19, out17, out20)



def litehrnet_fused(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    outs = litehrnet_fused_impl(in_0, in_1, in_2, in_3, in_4, in_5, in_6)
    return (outs[0], outs[1], outs[2], outs[3])



def replacement_func():
    return litehrnet_fused