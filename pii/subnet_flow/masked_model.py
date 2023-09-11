import einops
import torch
from jaxtyping import Float
from torch import nn
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint


def get_hook_points(model: nn.Module):
    for module in model.children():
        if isinstance(module, HookPoint):
            yield module
        else:
            yield from get_hook_points(module)


class PerTokenMaskedTransformer(nn.Module):
    def __init__(self, tl_model: HookedTransformer, base_prompt: str):
        """
        Warning: This will modify the transformer in-place with perma-hooks.
        It will clear existing perma-hooks in the places where it adds new ones.
        This is to make the operation idempotent.
        """
        super().__init__()

        self.tl_model = tl_model
        self.tl_model.requires_grad_(False)
        self.n_tokens = tl_model.to_tokens(base_prompt).shape[1]

        # Allows for disabling of masks
        self.masks_active = True

        mask_dict: dict[str, nn.Parameter] = {}
        for hp in get_hook_points(tl_model):
            include_patterns = ["attn.hook_z", "mlp_out"]
            if not any(pattern in hp.name for pattern in include_patterns):
                continue

            mask = nn.Parameter(
                torch.ones(
                    size=(self.n_tokens,),
                    dtype=torch.float32,
                    device=self.device,
                )
            )
            key = (
                hp.name.replace(".", "_")
                .replace("blocks_", "")
                .replace("hook_", "")
            )
            mask_dict[key] = mask

            def mask_hook(
                act: Float[torch.Tensor, "batch token head_index d_head"]
                | Float[torch.Tensor, "batch token d_model"],
                hook: HookPoint,  # Unused
            ):
                if not self.masks_active:
                    return act

                # Pad mask with 1s to match act
                mask_padded = torch.cat(
                    [
                        mask,
                        torch.ones(
                            size=(act.shape[1] - self.n_tokens,),
                            dtype=mask.dtype,
                            device=mask.device,
                        ),
                    ]
                )

                if "mlp_out" in hook.name:
                    return act * einops.rearrange(
                        mask_padded, "token -> 1 token 1"
                    )
                else:
                    return act * einops.rearrange(
                        mask_padded, "token -> 1 token 1 1"
                    )

            hp.remove_hooks(including_permanent=True)
            hp.add_perma_hook(hook=mask_hook)

        self.masks = nn.ParameterDict(mask_dict)

        # TODO: Support adding masks to attention patterns

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, *args, **kwargs):
        """Convenience method that calls out to the underlying tl_model."""
        return self.tl_model(*args, **kwargs)
