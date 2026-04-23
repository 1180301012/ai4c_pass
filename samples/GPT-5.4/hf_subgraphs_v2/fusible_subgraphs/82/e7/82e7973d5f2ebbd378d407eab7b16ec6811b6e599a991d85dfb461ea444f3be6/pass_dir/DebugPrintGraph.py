import torch


def pattern(in_0, in_1):
    return in_0


def replacement_args(in_0, in_1):
    return (in_0,)


@torch.fx.wrap
def identity_debug(x):
    return x


def replacement_func():
    return identity_debug


def _debug_pass(gm):
    print('[DEBUG GRAPH START]', flush=True)
    print(gm.graph, flush=True)
    print('[DEBUG GRAPH END]', flush=True)
    from torch.fx.passes.infra.pass_manager import PassResult
    return PassResult(gm, False)