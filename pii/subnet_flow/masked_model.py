import torch
from torch import nn
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint


def get_hook_points(model: nn.Module):
    for module in model.children():
        if isinstance(module, HookPoint):
            yield module
        else:
            yield from get_hook_points(module)


class MaskedTransformer(nn.Module):
    def __init__(self, transformer: HookedTransformer):
        """
        Warning: This will modify the transformer in-place with perma-hooks.
        It will clear existing perma-hooks in the places where it adds new ones.
        This is to make the operation idempotent.
        """
        super().__init__()

        self.transformer = transformer
        self.transformer.requires_grad_(False)
        transformer_device = next(transformer.parameters()).device

        # Allows for disabling of masks
        self.masks_active = True

        mask_dict: dict[str, nn.Parameter] = {}
        for hp in get_hook_points(transformer):
            include_patterns = ["attn_out", "mlp_out"]
            if not any(pattern in hp.name for pattern in include_patterns):
                continue

            param = nn.Parameter(
                torch.scalar_tensor(
                    1.0,
                    dtype=torch.float32,
                    device=transformer_device,
                )
            )
            key = (
                hp.name.replace(".", "_")
                .replace("blocks_", "")
                .replace("hook_", "")
            )
            mask_dict[key] = param
            hp.remove_hooks(including_permanent=True)
            hp.add_perma_hook(
                hook=lambda act, hook: act * param if self.masks_active else act
            )

        self.masks = nn.ParameterDict(mask_dict)

        # TODO: Support adding masks to attention patterns

    def forward(self, *args, **kwargs):
        """Convenience method that calls out to the underlying transformer."""
        return self.transformer(*args, **kwargs)
