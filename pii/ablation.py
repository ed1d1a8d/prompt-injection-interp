"""Code for running ablations on TransformerLens models."""
import functools
from typing import Iterable

import torch
import transformer_lens.utils as tl_utils
from jaxtyping import Float
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.hook_points import HookPoint


def run_with_ablation(
    tl_model: HookedTransformer,
    prompt: str,
    ablation_cache: ActivationCache | None = None,
    attn_head_locs: Iterable[tuple[int, int, int]] = (),  # (lyr, head, seq)
    attn_bias_locs: Iterable[tuple[int, int]] = (),  # (lyr, seq)
    mlp_head_locs: Iterable[tuple[int, int]] = (),  # (lyr, seq)
    embed_locs: Iterable[tuple[int]] = (),  # seq
    **kwargs,
):
    """
    Runs the tl_model with the specified heads ablated.
    If ablation_cache is None, will run with zeroed activations.

    Innefficient, but should be fine when number of ablations is small.

    TODO: Also support ablating position embeddings. We don't need this for
          llama2 models because they use rotary embeddings and thus don't have
          initial position embeddings.
    """
    # Deduplicate locations
    attn_head_locs = set(attn_head_locs)
    attn_bias_locs = set(attn_bias_locs)
    mlp_head_locs = set(mlp_head_locs)
    embed_locs = set(embed_locs)

    def ablate_attn_head(
        zs: Float[torch.Tensor, "b seq head d"],
        hook: HookPoint,
        layer: int,
    ):
        for lyr, head, seq in attn_head_locs:
            if layer == lyr:
                if ablation_cache is None:
                    zs[:, seq, head] = 0
                else:
                    if ablation_cache.has_batch_dim:
                        raise NotImplementedError
                    else:
                        zs[:, seq, head] = ablation_cache["z", lyr][
                            None, seq, head
                        ]
        return zs

    def ablate_attn_bias(
        act: Float[torch.Tensor, "b seq d"],
        hook: HookPoint,
        layer: int,
    ):
        for lyr, seq in attn_bias_locs:
            if layer == lyr:
                if ablation_cache is None:
                    act[:, seq] -= tl_model.b_O[lyr]
                else:
                    # noop because biases are the same for different runs
                    pass
        return act

    def ablate_mlp(
        act: Float[torch.Tensor, "b seq d"],
        hook: HookPoint,
        layer: int,
    ):
        for lyr, seq in mlp_head_locs:
            if layer == lyr:
                if ablation_cache is None:
                    act[:, seq] = 0
                else:
                    if ablation_cache.has_batch_dim:
                        raise NotImplementedError
                    else:
                        act[:, seq] = ablation_cache["mlp_out", lyr][None, seq]
        return act

    def ablate_embed(
        act: Float[torch.Tensor, "b seq d"],
        hook: HookPoint,
    ):
        for (seq,) in embed_locs:
            if ablation_cache is None:
                act[:, seq] = 0
            else:
                # noop because embeddings are the same for different runs
                pass
        return act

    return tl_model.run_with_hooks(
        prompt,
        fwd_hooks=[
            (
                tl_utils.get_act_name("z", l),
                functools.partial(ablate_attn_head, layer=l),
            )
            for l in set(l for l, _, _ in attn_head_locs)
        ]
        + [
            (
                tl_utils.get_act_name("attn_out", l),
                functools.partial(ablate_attn_bias, layer=l),
            )
            for l in set(l for l, _ in attn_bias_locs)
        ]
        + [
            (
                tl_utils.get_act_name("mlp_out", l),
                functools.partial(ablate_mlp, layer=l),
            )
            for l in set(l for l, _ in mlp_head_locs)
        ]
        + [(tl_utils.get_act_name("embed"), ablate_embed)],
        **kwargs,
    )
