import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    W_out = W + (B @ A) * alpha/r
    Where W is frozen, only A and B are trained.
    """

    def __init__(self, base_layer: nn.Linear, r: int, alpha: float) -> None:
        super().__init__()

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.base = base_layer
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        # A initialized to random N(0, 0.01); B initialized to 0 so BAx = 0 initially.
        self.A = nn.Parameter(torch.randn(r, base_layer.in_features, device=device, dtype=dtype) * 0.01)
        self.B = nn.Parameter(torch.zeros(base_layer.out_features, r, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor):
        base_projection = self.base(x)
        lora_projection_matrix = torch.matmul(self.B, self.A)  # (out_features, in_features)
        lora_projection = F.linear(x, lora_projection_matrix)
        return base_projection + self.scaling * lora_projection


def inject_lora(model, r, alpha):
    target_substrings = {"q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"}

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and any(key in name for key in target_substrings):
            parent = model
            components = name.split(".")
            for comp in components[:-1]:
                parent = getattr(parent, comp)

            layer_name = components[-1]
            original_layer = getattr(parent, layer_name)
            setattr(parent, layer_name, LoRALinear(original_layer, r=r, alpha=alpha))

    return model