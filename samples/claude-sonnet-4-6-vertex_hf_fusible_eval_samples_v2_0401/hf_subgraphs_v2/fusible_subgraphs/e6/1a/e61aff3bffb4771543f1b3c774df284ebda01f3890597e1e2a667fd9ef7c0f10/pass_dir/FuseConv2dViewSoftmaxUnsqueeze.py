import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Config pruner: softmax kernel needs BLOCK_SIZE >= n_cols for single-pass
# ---------------------------------------------------------------------------
def _prune_softmax_configs(configs, named_args, **kwargs):
    n_cols = named_args["n_cols"]
    valid = [c for c in configs if c.kwargs["BLOCK_SIZE"] >= n_cols]
    return valid if valid else [max(configs, key=lambda c: c.kwargs["BLOCK_SIZE"])]


# ---------------------------------------------------------------------------
# Triton kernel 1: Linear projection (1×1 conv with 1 output channel)
#   Computes: out[flat] = sum_c( in2[b,c,hw] * w[c] ) + bias
#   where flat = b*HW + hw, stored as float32 for precision.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SPATIAL": 128}),
        triton.Config({"BLOCK_SPATIAL": 256}),
        triton.Config({"BLOCK_SPATIAL": 512}),
        triton.Config({"BLOCK_SPATIAL": 1024}),
    ],
    key=["n_elem"],
)
@triton.jit
def _linear_proj_kernel(
    in2_ptr,         # [B, C, HW] contiguous NCHW
    w_ptr,           # [C] (weight[0, :, 0, 0])
    bias_ptr,        # [1]
    out_ptr,         # [n_elem] float32
    n_elem,          # B * HW
    C,               # number of input channels
    HW,              # H * W
    BLOCK_SPATIAL: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SPATIAL + tl.arange(0, BLOCK_SPATIAL)
    mask = offsets < n_elem

    b  = offsets // HW
    hw = offsets % HW

    acc = tl.zeros([BLOCK_SPATIAL], dtype=tl.float32)

    # Dynamic loop over channels
    for c in range(C):
        w_c = tl.load(w_ptr + c)
        inp_off = b * C * HW + c * HW + hw
        x_c = tl.load(in2_ptr + inp_off, mask=mask, other=0.0)
        acc = acc + x_c.to(tl.float32) * w_c.to(tl.float32)

    bias_val = tl.load(bias_ptr)
    acc = acc + bias_val.to(tl.float32)

    tl.store(out_ptr + offsets, acc, mask=mask)


# ---------------------------------------------------------------------------
# Triton kernel 2: Row-wise softmax
#   One program per row; BLOCK_SIZE must be >= n_cols (enforced by pruner).
#   Computation in fp32, output cast back to input dtype.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}),
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
        triton.Config({"BLOCK_SIZE": 2048}),
        triton.Config({"BLOCK_SIZE": 4096}),
    ],
    key=["n_cols"],
    prune_configs_by={"early_config_prune": _prune_softmax_configs},
)
@triton.jit
def _softmax_rows_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    base_in  = input_ptr  + row_idx * n_cols
    base_out = output_ptr + row_idx * n_cols

    offsets = tl.arange(0, BLOCK_SIZE)
    mask    = offsets < n_cols

    x     = tl.load(base_in + offsets, mask=mask, other=float("-inf"))
    x_f32 = x.to(tl.float32)

    x_max   = tl.max(tl.where(mask, x_f32, float("-inf")), axis=0)
    shifted = tl.where(mask, x_f32 - x_max, float("-inf"))
    exp_x   = tl.exp(shifted)
    exp_x   = tl.where(mask, exp_x, 0.0)
    sum_exp = tl.sum(exp_x, axis=0)

    out = (exp_x / sum_exp).to(x.dtype)
    tl.store(base_out + offsets, out, mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper – fully Triton-based, no blocked torch.conv2d call
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _fused_conv2d_view_softmax_unsqueeze(in_0, in_1, in_2, d0, d1, d2):
    """
    Triton replacement for:
        conv2d  = torch.conv2d(in_2, in_1, in_0, (1,1), (0,0), (1,1), 1)
        tmp_3   = conv2d.view(d0, d1, d2)
        tmp_4   = softmax(tmp_3, dim=2)
        tmp_5   = tmp_4.unsqueeze(-1)
        return (tmp_5,)

    Pipeline:
      1. Triton linear-projection kernel → float32 intermediate [B*H*W]
      2. Cast back to original dtype, reshape → [d0*d1, d2]
      3. Triton softmax kernel (row-wise)
      4. Reshape → [d0, d1, d2, 1]
    """
    orig_dtype = in_2.dtype
    B  = in_2.shape[0]
    C  = in_2.shape[1]
    H  = in_2.shape[2]
    W  = in_2.shape[3]
    HW     = H * W
    n_elem = B * HW

    in2_c  = in_2.contiguous()
    w_flat = in_1.reshape(C).contiguous()   # [1,C,1,1] → [C]

    # Step 1 – linear projection in float32
    proj = torch.empty(n_elem, dtype=torch.float32, device=in_2.device)
    grid_proj = lambda meta: (triton.cdiv(n_elem, meta["BLOCK_SPATIAL"]),)
    _linear_proj_kernel[grid_proj](in2_c, w_flat, in_0, proj, n_elem, C, HW)

    # Step 2 – cast & reshape for softmax
    n_rows = d0 * d1
    n_cols = d2
    flat   = proj.to(orig_dtype).reshape(n_rows, n_cols).contiguous()
    output = torch.empty_like(flat)

    # Step 3 – Triton softmax
    _softmax_rows_kernel[(n_rows,)](flat, output, n_rows, n_cols)

    # Step 4 – final shape [d0, d1, d2, 1]
    return (output.view(d0, d1, d2, 1),)


# ---------------------------------------------------------------------------
# Pattern / replacement hooks required by the AI4C framework
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, d0, d1, d2):
    """
    Matches:
        conv2d = torch.conv2d(in_2, in_1, in_0, (1,1), (0,0), (1,1), 1)
        tmp_3  = conv2d.view(d0, d1, d2)
        tmp_4  = torch.nn.functional.softmax(tmp_3, 2, _stacklevel=5)
        tmp_5  = tmp_4.unsqueeze(-1)
        return (tmp_5,)
    """
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3  = conv2d.view(d0, d1, d2)
    tmp_4  = torch.nn.functional.softmax(tmp_3, 2, _stacklevel=5)
    tmp_5  = tmp_4.unsqueeze(-1)
    return (tmp_5,)


def replacement_args(in_0, in_1, in_2, d0, d1, d2):
    return (in_0, in_1, in_2, d0, d1, d2)


def replacement_func():
    return _fused_conv2d_view_softmax_unsqueeze