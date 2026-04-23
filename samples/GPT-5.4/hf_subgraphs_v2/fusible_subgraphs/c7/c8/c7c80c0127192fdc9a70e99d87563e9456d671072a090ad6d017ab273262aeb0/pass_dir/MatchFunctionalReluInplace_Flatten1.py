import torch
from pass_dir.relu_shared import relu_dispatch
from graph_net_bench.torch.backend import pass_mgr_backend as _pmb

if not hasattr(_pmb.PatternReplacementPass, "_ai4c_debug_graph_patch"):
    _orig_diag = _pmb.PatternReplacementPass._print_diagnostic_report
    def _debug_diag(self, gm):
        print("[AI4C_DEBUG_GRAPH]", gm.graph, flush=True)
        return _orig_diag(self, gm)
    _pmb.PatternReplacementPass._print_diagnostic_report = _debug_diag
    _pmb.PatternReplacementPass._ai4c_debug_graph_patch = True


def pattern(in_0):
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = tmp_0.flatten(1, -1)
    return tmp_1


def replacement_args(in_0):
    return (in_0, "inplace")


def replacement_func():
    return relu_dispatch