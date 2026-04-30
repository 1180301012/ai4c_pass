import torch
import triton
import triton.language as tl
import operator
import inspect
import torch.fx


@triton.jit
def add2_kernel(in0_ptr, in1_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x0 = tl.load(in0_ptr + offs, mask=mask)
    x1 = tl.load(in1_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x0 + x1, mask=mask)


@triton.jit
def add3_kernel(in0_ptr, in1_ptr, in2_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x0 = tl.load(in0_ptr + offs, mask=mask)
    x1 = tl.load(in1_ptr + offs, mask=mask)
    x2 = tl.load(in2_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x0 + x1 + x2, mask=mask)


def _do_identity(in_0):
    return in_0


def _do_add2(in_0, in_1):
    n_elements = in_0.numel()
    out = torch.empty_like(in_0)
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    add2_kernel[grid](in_0, in_1, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    return out


def _do_add3(in_0, in_1, in_2):
    n_elements = in_0.numel()
    out = torch.empty_like(in_0)
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    add3_kernel[grid](in_0, in_1, in_2, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    return out


@torch.fx.wrap
def fused_add_dispatch(*args):
    route = args[-1]
    if route == "identity":
        return _do_identity(args[0])
    elif route == "add2":
        return _do_add2(args[0], args[1])
    elif route == "add3_v1":
        return _do_add3(args[0], args[1], args[2])
    elif route == "add3_v2":
        return _do_add3(args[0], args[1], args[2])
    else:
        return _do_identity(args[0])


# ============ Pre-built pattern GraphModules (without mean) ============

def _build_pattern_identity():
    """Pattern: 0 + in_0; iadd 0 → returns iadd result"""
    graph = torch.fx.Graph()
    in_0 = graph.placeholder('in_0')
    add_node = graph.call_function(operator.add, (0, in_0))
    iadd_node = graph.call_function(operator.iadd, (add_node, 0))
    graph.output((iadd_node,))

    class M(torch.nn.Module):
        pass

    gm = torch.fx.GraphModule(M(), graph)
    gm.__signature__ = inspect.Signature([
        inspect.Parameter('in_0', inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ])
    return gm


def _build_pattern_add2():
    """Pattern: 0 + in_1; iadd in_0 → returns iadd result"""
    graph = torch.fx.Graph()
    in_0 = graph.placeholder('in_0')
    in_1 = graph.placeholder('in_1')
    add_node = graph.call_function(operator.add, (0, in_1))
    iadd_node = graph.call_function(operator.iadd, (add_node, in_0))
    graph.output((iadd_node,))

    class M(torch.nn.Module):
        pass

    gm = torch.fx.GraphModule(M(), graph)
    gm.__signature__ = inspect.Signature([
        inspect.Parameter('in_0', inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter('in_1', inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ])
    return gm


def _build_pattern_add3_v1():
    """Pattern: in_1 + in_2; iadd in_0 → returns iadd result"""
    graph = torch.fx.Graph()
    in_0 = graph.placeholder('in_0')
    in_1 = graph.placeholder('in_1')
    in_2 = graph.placeholder('in_2')
    add_node = graph.call_function(operator.add, (in_1, in_2))
    iadd_node = graph.call_function(operator.iadd, (add_node, in_0))
    graph.output((iadd_node,))

    class M(torch.nn.Module):
        pass

    gm = torch.fx.GraphModule(M(), graph)
    gm.__signature__ = inspect.Signature([
        inspect.Parameter('in_0', inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter('in_1', inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter('in_2', inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ])
    return gm


def _build_pattern_add3_v2():
    """Pattern: in_0 + in_1; iadd in_2 → returns iadd result"""
    graph = torch.fx.Graph()
    in_0 = graph.placeholder('in_0')
    in_1 = graph.placeholder('in_1')
    in_2 = graph.placeholder('in_2')
    add_node = graph.call_function(operator.add, (in_0, in_1))
    iadd_node = graph.call_function(operator.iadd, (add_node, in_2))
    graph.output((iadd_node,))

    class M(torch.nn.Module):
        pass

    gm = torch.fx.GraphModule(M(), graph)
    gm.__signature__ = inspect.Signature([
        inspect.Parameter('in_0', inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter('in_1', inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter('in_2', inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ])
    return gm


pattern_identity = _build_pattern_identity()
pattern_add2 = _build_pattern_add2()
pattern_add3_v1 = _build_pattern_add3_v1()
pattern_add3_v2 = _build_pattern_add3_v2()