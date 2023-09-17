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
        noise_std: float = 0.0,
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
        self.noise_std = noise_std

        self.masks = nn.ParameterDict()
        for hp in get_hook_points(tl_model):
            include_patterns = [
                "attn.hook_z",
                "attn_out",
                "pattern",
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
            elif "pattern" in hp.name:
                self.masks[self.get_mask_key(hp)] = nn.Parameter(
                    torch.tril(
                        self.get_ones_vector(
                            self.tl_model.cfg.n_heads,
                            self.n_tokens - self.mask_start_idx,
                            self.n_tokens - self.mask_start_idx,
                        )
                    ),
                )
            elif "attn_out" in hp.name or "mlp_out" in hp.name:
                self.masks[self.get_mask_key(hp)] = nn.Parameter(
                    self.get_ones_vector(self.n_tokens - self.mask_start_idx),
                )
            else:
                continue

            def mask_hook(
                act: Float[torch.Tensor, "batch token head_index d_head"]
                | Float[torch.Tensor, "batch head_index token token"]
                | Float[torch.Tensor, "batch token d_model"],
                hook: HookPoint,
            ):
                if not self.masks_active:
                    return act

                mask = self.get_mask(hook).clone()
                mask = einops.rearrange(mask, "... -> 1 ...")  # Add batch dim

                # Note that the one_threshold and zero_threshold are inclusive,
                # meaning gradient descent is sticky at the threshold.
                mask[mask >= self.one_threshold] = 1
                mask[mask <= self.zero_threshold] = 0

                if self.noise_std > 0 and self.training:
                    mask = einops.repeat(
                        mask, "1 ... -> b ...", b=act.shape[0]
                    ).clone()

                    noise = torch.clamp(
                        torch.randn_like(mask) * self.noise_std * (mask > 0),
                        min=-mask / 2,
                        max=(1 - mask) / 2,
                    )
                    mask += noise

                if "attn.hook_z" in hook.name:
                    act[
                        :, self.mask_start_idx : self.n_tokens, :, :
                    ] *= einops.rearrange(mask, "b tok head -> b tok head 1")
                elif "pattern" in hook.name:
                    # Needed to avoid in-place modification so that backprop
                    # works.
                    act = act.clone()

                    act[
                        :,
                        :,
                        self.mask_start_idx : self.n_tokens,
                        self.mask_start_idx : self.n_tokens,
                    ] *= mask

                    act_tot = act.sum(dim=-1, keepdim=True)
                    act /= torch.maximum(
                        act_tot, torch.tensor(1e-9, device=act.device)
                    )
                    act[
                        :,
                        :,
                        self.mask_start_idx : self.n_tokens,
                        self.mask_start_idx : self.n_tokens,
                    ] *= mask.max(dim=-1, keepdim=True)[0]
                else:
                    act[
                        :, self.mask_start_idx : self.n_tokens, :
                    ] *= einops.rearrange(mask, "b tok -> b tok 1")

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
        )

    def get_mask(self, hp: HookPoint):
        return self.masks[self.get_mask_key(hp)]

    def forward(self, *args, **kwargs):
        """Convenience method that calls out to the underlying tl_model."""
        return self.tl_model(*args, **kwargs)
