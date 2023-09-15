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

        self.zero_threshold = 0.0  # If smaller than 0, has no effect.
        # A 0 value makes the mask sticky at 0.
        self.one_threshold = 1.1  # If larger than 1, has no effect

        self.masks = nn.ParameterDict()
        for hp in get_hook_points(tl_model):
            include_patterns = [
                "attn.hook_z",
                "attn_out",
                # "attn_scores",
                "mlp_out",
            ]
            if not any(pattern in hp.name for pattern in include_patterns):
                continue

            if "attn.hook_z" in hp.name:
                self.masks[self.get_mask_key(hp)] = nn.Parameter(
                    self.get_ones_vector(
                        self.n_tokens - self.mask_start_idx,
                        self.tl_model.cfg.n_heads,
                    ),
                )
            elif "attn_scores" in hp.name:
                raise NotImplementedError
                self.masks[self.get_mask_key(hp)] = nn.Parameter(
                    self.get_ones_vector(
                        self.tl_model.cfg.n_heads,
                        self.n_tokens - self.mask_start_idx,
                        self.n_tokens - self.mask_start_idx,
                    ),
                )
            else:
                self.masks[self.get_mask_key(hp)] = nn.Parameter(
                    self.get_ones_vector(self.n_tokens - self.mask_start_idx),
                )

            def mask_hook(
                act: Float[torch.Tensor, "batch token head_index d_head"]
                | Float[torch.Tensor, "batch token d_model"]
                | Float[torch.Tensor, "batch token token"],
                hook: HookPoint,
            ):
                if not self.masks_active:
                    return act

                mask = self.get_mask(hook).clone()

                # Note that the one_threshold and zero_threshold are inclusive,
                # meaning gradient descent is sticky at the threshold.
                mask[mask >= self.one_threshold] = 1
                mask[mask <= self.zero_threshold] = 0

                if "attn.hook_z" in hook.name:
                    act[
                        :, self.mask_start_idx : self.n_tokens, :, :
                    ] *= einops.rearrange(mask, "... -> 1 ... 1")
                elif "attn_scores" in hook.name:
                    raise NotImplementedError
                else:
                    act[
                        :, self.mask_start_idx : self.n_tokens, :
                    ] *= einops.rearrange(mask, "... -> 1 ... 1")

                return act

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
            .replace("attn_scores", "scores")
        )

    def get_mask(self, hp: HookPoint):
        return self.masks[self.get_mask_key(hp)]

    def forward(self, *args, **kwargs):
        """Convenience method that calls out to the underlying tl_model."""
        return self.tl_model(*args, **kwargs)
