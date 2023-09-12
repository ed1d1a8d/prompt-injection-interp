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
    def __init__(
        self,
        tl_model: HookedTransformer,
        base_prompt: str,
        mask_start_idx: int = 0,
    ):
        """
        Warning: This will modify the transformer in-place with perma-hooks.
        It will clear existing perma-hooks in the places where it adds new ones.
        This is to make the operation idempotent.
        """
        super().__init__()

        self.tl_model = tl_model
        self.tl_model.requires_grad_(False)
        self.n_tokens = tl_model.to_tokens(base_prompt).shape[1]
        self.mask_start_idx = mask_start_idx

        # Allows for disabling of masks
        self.masks_active = True

        self.zero_threshold = 0.0

        self.masks = nn.ParameterDict()
        for hp in get_hook_points(tl_model):
            # include_patterns = ["attn.hook_z", "mlp_out"]
            include_patterns = ["mlp_out", "attn_out"]
            if not any(pattern in hp.name for pattern in include_patterns):
                continue

            self.masks[self.get_mask_key(hp)] = nn.Parameter(
                self.get_ones_vector(self.n_tokens - self.mask_start_idx),
            )

            def mask_hook(
                act: Float[torch.Tensor, "batch token head_index d_head"]
                | Float[torch.Tensor, "batch token d_model"],
                hook: HookPoint,
            ):
                if not self.masks_active:
                    return act

                mask = self.get_mask(hook)

                # Pad mask with 1s to match act
                mask_padded = torch.cat(
                    [
                        self.get_ones_vector(self.mask_start_idx),
                        mask,
                        self.get_ones_vector(
                            max(0, act.shape[1] - self.n_tokens)
                        ),
                    ]
                )

                mask_reshaped = einops.rearrange(
                    mask_padded, "token -> 1 token"
                )
                while mask_reshaped.ndim < act.ndim:
                    mask_reshaped = einops.rearrange(
                        mask_reshaped, "... -> ... 1"
                    )

                return (
                    act * mask_reshaped * (mask_reshaped > self.zero_threshold)
                )

            hp.remove_hooks(including_permanent=True)
            hp.add_perma_hook(hook=mask_hook)

        # TODO: Support adding masks to attention patterns

    def get_ones_vector(self, n: int):
        return torch.ones(size=(n,), dtype=torch.float32, device=self.device)

    @property
    def device(self):
        return next(self.parameters()).device

    def get_mask_key(self, hp: HookPoint):
        return (
            hp.name.replace(".", "_")
            .replace("blocks_", "")
            .replace("hook_", "")
        )

    def get_mask(self, hp: HookPoint):
        return self.masks[self.get_mask_key(hp)]

    def forward(self, *args, **kwargs):
        """Convenience method that calls out to the underlying tl_model."""
        return self.tl_model(*args, **kwargs)
