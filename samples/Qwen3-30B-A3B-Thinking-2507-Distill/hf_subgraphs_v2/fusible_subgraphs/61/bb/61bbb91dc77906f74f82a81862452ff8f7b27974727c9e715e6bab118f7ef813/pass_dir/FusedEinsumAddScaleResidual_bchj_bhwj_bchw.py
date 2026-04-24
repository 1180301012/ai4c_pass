import torch
import triton
import triton.language as tl
import sys

# Print the graph when module is loaded to understand the compiled graph structure
def _dump_graph():
    try:
        import importlib.util, graph_net_bench.torch.test_compiler as _tc
        spec = importlib.util.spec_from_file_location(
            "dbg_model",
            f"{os.environ.get('SAMPLE_ROOT', '.')}/graphs/hf_subgraphs_v2/fusible_subgraphs/bfloat16/7/samples/mmseg/CCNet_R101/_decomposed/CCNet_R101_start388_end393_9/model.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        model_cls = mod.GraphModule
        model = model_cls()
        sys.path.insert(0, os.environ.get('ai4c_repo_root', '.'))
        import torch.fx as _fx
        traced = _fx.symbolic_trace(model)
        with open('/tmp/fx_graph.txt', 'w') as f:
            f.write(str(traced.graph))
        sys.stderr.write(f"[AI4C] Graph dumped to /tmp/fx_graph.txt\n")
    except Exception as e:
        sys.stderr.write(f"[AI4C] Dump failed: {e}\n")

_dump_graph()


# ──────────────────────────────────────────────────────────────────────────────
# Triton kernel: fuses the full chain
#   result = (in_3 + einsum(in_4,in_1)) * scale + in_2
# ──────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_BC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_BC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_BC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_BC': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_BC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_BC': 64}, num_warps=4, num_stages=2),
    ],
    key=['BC', 'H', 'W'],
)
@triton.jit
def _fused_einsum_add_scale_residual_kernel(
    in4_ptr, in1_ptr, in3_ptr, in2_ptr, in0_ptr, out_ptr,
    BC, H, W,
    J: tl.constexpr,
    C: tl.constexpr,
    bc_s, h_s, w_s,
    BLOCK_BC: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_h  = tl.program_id(1)

    bc0    = pid_bc * BLOCK_BC
    bc_off = bc0 + tl.arange(0, BLOCK_BC)
    bc_mask = bc_off < BC
    j = tl.arange(0, J)
    w = tl.arange(0, W)

    # b-index: b = bc // C
    C1_base = H * W * J // C
    b_off   = bc_off // C
    b_base  = b_off * C1_base

    # a = in_4[bc, h, :J]
    in4_off = b_base + pid_h * J + j[None, :]
    a = tl.load(in4_ptr + in4_off, mask=bc_mask[:, None], other=0.0)

    # b = in_1[b, h, :W, :J]
    in1_off = b_base[:, None] + pid_h * (W * J) + w[None, :] * J + j[None, :]
    b = tl.load(in1_ptr + in1_off, mask=bc_mask[:, None], other=0.0)

    # acc = a @ tl.trans(b,1,2) = [BLOCK_BC,J] @ [J,W]
    b_t = tl.trans(b, 1, 2)
    acc = tl.dot(a, b_t, out_dtype=tl.float32)

    scale = tl.load(in0_ptr).to(tl.float32)
    out_off = bc_off[:, None] * bc_s + pid_h * h_s + w[None, :] * w_s
    in3_v = tl.load(in3_ptr + out_off, mask=bc_mask[:, None], other=0.0).to(tl.float32)
    in2_v = tl.load(in2_ptr + out_off, mask=bc_mask[:, None], other=0.0).to(tl.float32)

    result = acc * scale + in3_v + in2_v
    tl.store(out_ptr + out_off, result, mask=bc_mask[:, None])


@torch.fx.wrap
def _fused_einsum_add_scale_residual(in_0, in_1, in_2, in_3, in_4):
    B   = in_4.shape[0]
    C   = in_4.shape[1]
    H   = in_4.shape[2]
    J   = in_4.shape[3]
    W   = in_1.shape[2]
    BC  = B * C

    out = torch.empty_like(in_3)

    def grid(meta):
        return ((BC + meta['BLOCK_BC'] - 1) // meta['BLOCK_BC'], H)

    _fused_einsum_add_scale_residual_kernel[grid](
        in_4, in_1, in_3, in_2, in_0, out,
        BC, H, W, J, C,
        out.stride(0), out.stride(2), out.stride(3),
    )
    return out


def replacement_func():
    return _fused_einsum_add_scale_residual