import torch
import triton
import triton.language as tl
import torch.fx


# Build an FX pattern graph manually so call args/kwargs exactly match the target graph.
def _build_pattern_graph_module():
    g = torch.fx.Graph()
    in_0 = g.placeholder("in_0")
    in_1 = g.placeholder("in_1")
    in_2 = g.placeholder("in_2")
    conv2d = g.call_function(torch.conv2d, args=(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1), kwargs={})
    tmp_3 = g.call_function(torch.nn.functional.silu, args=(conv2d,), kwargs={"inplace": False})
    tmp_4 = g.call_function(torch.nn.functional.dropout, args=(tmp_3, 0.0, False, False), kwargs={})
    g.output((tmp_4,))
    return torch.fx.GraphModule(torch.nn.Module(), g)


# Pattern object used by the pass manager.
pattern = _build_pattern_graph_module()


# Extract arguments for replacement
# Original forward order is (bias, weight, input)
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=3),
    ],
    key=["M", "C_OUT", "C_IN"],
)
@triton.jit
def _fused_conv1x1_bias_silu_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    M,
    H,
    W,
    C_IN,
    C_OUT,
    stride_xn,
    stride_xc,
    stride_xh,
    stride_xw,
    stride_wo,
    stride_wi,
    stride_on,
    stride_oc,
    stride_oh,
    stride_ow,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(C_OUT, BLOCK_N)

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    hw = H * W
    batch_idx = offs_m // hw
    spatial_idx = offs_m % hw
    h_idx = spatial_idx // W
    w_idx = spatial_idx % W

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, C_IN, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        x_ptrs = (
            x_ptr
            + batch_idx[:, None] * stride_xn
            + offs_k[None, :] * stride_xc
            + h_idx[:, None] * stride_xh
            + w_idx[:, None] * stride_xw
        )
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < C_IN)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        w_ptrs = w_ptr + offs_k[:, None] * stride_wi + offs_n[None, :] * stride_wo
        w_mask = (offs_k[:, None] < C_IN) & (offs_n[None, :] < C_OUT)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.dot(x, w)

    bias = tl.load(b_ptr + offs_n, mask=offs_n < C_OUT, other=0.0).to(tl.float32)
    acc += bias[None, :]

    sig = 1.0 / (1.0 + tl.exp(-acc))
    out = acc * sig

    out_ptrs = (
        out_ptr
        + batch_idx[:, None] * stride_on
        + offs_n[None, :] * stride_oc
        + h_idx[:, None] * stride_oh
        + w_idx[:, None] * stride_ow
    )
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < C_OUT)
    tl.store(out_ptrs, out, mask=out_mask)


@torch.fx.wrap
def fused_conv1x1_bias_silu_dropout0(bias, weight, x):
    batch = x.shape[0]
    h = x.shape[2]
    w = x.shape[3]
    c_out = weight.shape[0]
    c_in = x.shape[1]

    out = torch.empty((batch, c_out, h, w), device=x.device, dtype=x.dtype)
    m = batch * h * w

    grid = lambda META: (
        triton.cdiv(m, META["BLOCK_M"]) * triton.cdiv(c_out, META["BLOCK_N"]),
    )

    _fused_conv1x1_bias_silu_kernel[grid](
        x,
        weight,
        bias,
        out,
        m,
        h,
        w,
        c_in,
        c_out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
    )
    return (out,)


# Replacement function must return the callable itself

def replacement_func():
    return fused_conv1x1_bias_silu_dropout0