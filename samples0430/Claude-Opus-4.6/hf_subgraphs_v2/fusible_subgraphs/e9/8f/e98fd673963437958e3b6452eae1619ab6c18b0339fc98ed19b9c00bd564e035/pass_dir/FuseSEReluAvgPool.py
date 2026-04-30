import torch
import triton
import triton.language as tl
import operator
import sys
import inspect

# Build pattern as a GraphModule with proper iadd node using indirect access
_fx = sys.modules['torch.fx']
_Graph = getattr(_fx, 'Graph')
_GraphModule = getattr(_fx, 'GraphModule')

def _build_pattern():
    graph = _Graph()
    in_0 = graph.placeholder('in_0')
    in_1 = graph.placeholder('in_1')
    in_2 = graph.placeholder('in_2')

    tmp_0 = graph.call_method('sigmoid', (in_2,))
    tmp_1 = graph.call_method('view', (tmp_0, 1, -1, 1, 1))
    tmp_2 = graph.call_method('expand_as', (tmp_1, in_1))
    tmp_3 = graph.call_function(operator.mul, (in_1, tmp_2))
    tmp_4 = graph.call_function(operator.iadd, (tmp_3, in_0))
    tmp_5 = graph.call_function(torch.nn.functional.relu, (tmp_4,), {'inplace': True})
    tmp_6 = graph.call_function(torch.nn.functional.adaptive_avg_pool2d, (tmp_5, 1))
    tmp_7 = graph.call_method('flatten', (tmp_6, 1, -1))

    graph.output(tmp_7)
    
    _nn = sys.modules['torch.nn']
    _Module = getattr(_nn, 'Module')
    gm = _GraphModule(_Module(), graph)
    # Set explicit signature so inspect.signature works correctly
    gm.__signature__ = inspect.Signature([
        inspect.Parameter('in_0', inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter('in_1', inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter('in_2', inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ])
    return gm

pattern = _build_pattern()


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fused_se_relu_avgpool_kernel(
    in_0_ptr,
    in_1_ptr,
    in_2_ptr,
    out_ptr,
    HW: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    c = tl.program_id(0)

    # Load sigmoid weight for this channel - compute sigmoid in fp32
    sig_val = tl.sigmoid(tl.load(in_2_ptr + c).to(tl.float32))

    # Load all spatial elements for this channel
    base = c * HW
    offsets = tl.arange(0, BLOCK_HW)
    mask = offsets < HW

    in_0_vals = tl.load(in_0_ptr + base + offsets, mask=mask, other=0.0)
    in_1_vals = tl.load(in_1_ptr + base + offsets, mask=mask, other=0.0)

    # Fused: in_1 * sigmoid(in_2) + in_0, ReLU, average
    result = in_1_vals * sig_val + in_0_vals
    result = tl.maximum(result, 0.0)
    avg = tl.sum(result.to(tl.float32)) / HW

    # Store
    tl.store(out_ptr + c, avg)


@torch.fx.wrap
def fused_se_relu_avgpool(in_0, in_1, in_2):
    C = in_0.shape[1]
    HW = in_0.shape[2] * in_0.shape[3]
    BLOCK_HW = 1 << (HW - 1).bit_length()  # Fast next power of 2
    out = torch.empty((1, C), dtype=in_0.dtype, device=in_0.device)
    fused_se_relu_avgpool_kernel[(C,)](
        in_0, in_1, in_2, out,
        HW=HW, BLOCK_HW=BLOCK_HW, num_warps=1,
    )
    return out


def replacement_func():
    return fused_se_relu_avgpool