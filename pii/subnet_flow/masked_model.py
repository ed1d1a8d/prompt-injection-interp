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
            include_patterns = ["attn.hook_z", "attn_out", "mlp_out"]
            if not any(pattern in hp.name for pattern in include_patterns):
                continue

            if "attn.hook_z" in hp.name:
                self.masks[self.get_mask_key(hp)] = nn.Parameter(
                    self.get_ones_vector(
                        self.n_tokens - self.mask_start_idx,
                        self.tl_model.cfg.n_heads,
                    ),
                )
            else:
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

                mask_padded = self.get_padded_mask(hook, act.shape[1])
                mask_reshaped = einops.rearrange(mask_padded, "... -> 1 ... 1")

                return (
                    act * mask_reshaped * (mask_reshaped > self.zero_threshold)
                )

            hp.remove_hooks(including_permanent=True)
            hp.add_perma_hook(hook=mask_hook)

        # TODO: Support adding masks to attention patterns

    @property
    def device(self):
        return next(self.parameters()).device

    def get_ones_vector(self, *size: int):
        return torch.ones(size=size, dtype=torch.float32, device=self.device)

    def get_mask_key(self, hp: HookPoint):
        return (
            hp.name.replace(".", "_")
            .replace("blocks_", "")
            .replace("hook_", "")
        )

    def get_padded_mask(self, hp: HookPoint, target_len: int):
        assert target_len >= self.n_tokens

        mask = self.masks[self.get_mask_key(hp)]

        mask_shape = mask.shape
        prefix_shape = (self.mask_start_idx,) + mask_shape[1:]
        suffix_shape = (target_len - self.n_tokens,) + mask_shape[1:]

        return torch.cat(
            [
                self.get_ones_vector(*prefix_shape),
                mask,
                self.get_ones_vector(*suffix_shape),
            ],
            dim=0,
        )

    def forward(self, *args, **kwargs):
        """Convenience method that calls out to the underlying tl_model."""
        return self.tl_model(*args, **kwargs)
