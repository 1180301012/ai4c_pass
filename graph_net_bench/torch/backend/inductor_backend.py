import torch
from .graph_compiler_backend import GraphCompilerBackend


class InductorBackend(GraphCompilerBackend):
    def __init__(self, config):
        super().__init__(config)
        

    def __call__(self, model):
        mode = self.config.get("inductor_mode", "default")
        assert mode in [
            "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"
        ], f"Invalid mode: {mode}"
        return torch.compile(model, backend="inductor", mode=mode)

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()