"""
Helper functions for decomposing transformer activations into subcomponents.
"""
import dataclasses

import torch
from jaxtyping import Float
from transformer_lens import ActivationCache, HookedTransformer


@dataclasses.dataclass
class LabelledComponents:
    labels: list[str]
    components: Float[torch.Tensor, "n_components d"]

    def append_zero(self):
        return LabelledComponents(
            labels=self.labels + ["None"],
            components=torch.cat(
                [
                    self.components,
                    torch.zeros(
                        (1, self.components.shape[1]),
                        device=self.components.device,
                    ),
                ],
                dim=0,
            ),
        )


def get_attn_head_resid_components(
    tl_model: HookedTransformer,
    cache: ActivationCache,
    layer: int,
    pos: int,
) -> LabelledComponents:
    if "blocks.0.attn.hook_result" not in cache:
        cache.compute_head_results()

    n_heads = cache["result", layer][pos].shape[0]

    comp_labels = ["EMBED"]
    all_comps_list = [cache["embed"][pos][None, :]]
    for lyr in range(layer):
        comp_labels.extend([f"L{lyr}H{head}ATN" for head in range(n_heads)])
        all_comps_list.append(cache["result", lyr][pos])

        assert tl_model.b_O[layer].abs().max() == 0, "TODO: handle bias"

        comp_labels.append(f"L{lyr}MLP")
        all_comps_list.append(cache["mlp_out", lyr][pos][None, :])

    all_comps = torch.cat(all_comps_list, dim=0)
    assert torch.allclose(
        all_comps.sum(dim=0), cache["resid_pre", layer][pos], atol=1e-5
    )

    return LabelledComponents(
        labels=comp_labels,
        components=all_comps,
    )


def get_mlp_head_components(
    layer: int,
    pos: int,
    cache: ActivationCache,
    tl_model: HookedTransformer,
) -> LabelledComponents:
    if "blocks.0.attn.hook_result" not in cache:
        cache.compute_head_results()

    n_heads = cache["result", layer][pos].shape[0]

    comp_labels = ["EMBED"]
    all_comps_list = [cache["embed"][pos][None, :]]
    for lyr in range(layer + 1):
        comp_labels.extend([f"L{lyr}H{head}ATN" for head in range(n_heads)])
        all_comps_list.append(cache["result", lyr][pos])

        assert tl_model.b_O[layer].abs().max() == 0, "TODO: handle bias"

        if lyr == layer:
            break

    all_comps = torch.cat(all_comps_list, dim=0)
    assert torch.allclose(
        all_comps.sum(dim=0), cache["resid_mid", layer][pos], atol=1e-5
    )

    return LabelledComponents(
        labels=comp_labels,
        components=all_comps,
    )
