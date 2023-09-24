import functools
from typing import Iterable

import numpy as np
import plotly.graph_objects as go
import torch
import transformer_lens.utils as tl_utils
from jaxtyping import Float
from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.hook_points import HookPoint


def device(tl_model: HookedTransformer):
    return tl_model.b_O.device


def get_top_responses(
    prompt: str,
    model: HookedTransformer,
    top_k: int = 5,
    n_continuation_tokens: int = 5,
    prepend_bos: bool | None = None,
    print_prompt: bool = False,
    use_kv_cache: bool = True,
) -> tuple[int, str]:
    """
    Prints the most likely responses to a prompt.
    Adapted from transformer_lens.utils.test_prompt.
    """

    prompt_tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
    if print_prompt:
        print(
            "Tokenized prompt:", [model.to_string(t) for t in prompt_tokens[0]]
        )

    with torch.no_grad():
        logits = tl_utils.remove_batch_dim(model(prompt_tokens))[-1]
        probs = logits.softmax(dim=-1)

    _, sort_idxs = probs.sort(descending=True)
    for i in range(top_k):
        logit = logits[sort_idxs[i]]
        prob = probs[sort_idxs[i]]

        # Compute the continuation tokens
        continuation_tokens = model.generate(
            torch.concatenate(
                [
                    prompt_tokens,
                    torch.tensor(
                        [[sort_idxs[i]]],
                        device=prompt_tokens.device,
                    ),
                ],
                dim=1,
            ),
            max_new_tokens=n_continuation_tokens,
            prepend_bos=prepend_bos,
            verbose=False,
            temperature=0,
            use_past_kv_cache=use_kv_cache,
        )[0][prompt_tokens.shape[1] :]

        print(
            f"Rank {i}. "
            f"Logit: {logit:5.2f} "
            f"Prob: {prob:6.2%} "
            f"Tokens: ({continuation_tokens[0]:5d}) |{'|'.join([model.to_string(t) for t in continuation_tokens])}|"
        )

    # Return top token
    return sort_idxs[0].item(), model.to_string(sort_idxs[0])


def plot_head_data(
    lines: list[tuple[str, torch.Tensor | np.ndarray, list[str]]],
    annotation_text: str | None = None,
    **kwargs,
):
    fig = go.Figure()
    for name, xs, labels in lines:
        if isinstance(xs, torch.Tensor):
            xs = xs.flatten().cpu().numpy()
        else:
            xs = xs.flatten()
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=xs,
                mode="lines",
                name=name,
                opacity=0.7,
            )
        )

    labels = lines[0][2]
    spacing = 4 * 32
    fig.update_layout(
        xaxis_title="Layers and Heads",
        xaxis=dict(
            tickvals=[i for i in range(0, len(labels), spacing)],
            ticktext=labels[::spacing],
        ),
        hovermode="x unified",
        showlegend=True,
        **kwargs,
    )
    if annotation_text:
        fig.update_layout(
            annotations=[
                dict(
                    x=1,
                    y=1.05,
                    xref="paper",
                    yref="paper",
                    text=annotation_text,
                    showarrow=False,
                    font=dict(size=11),
                )
            ],
        )

    # Zoom out of plot slightly along x axis
    fig.update_xaxes(range=[-0.03 * len(labels), 1.03 * len(labels)])

    return fig


def run_with_ablation(
    tl_model: HookedTransformer,
    prompt: str,
    ablation_cache: ActivationCache | None = None,
    attn_head_locs: Iterable[tuple[int, int, int]] = (),  # (lyr, head, seq)
    attn_bias_locs: Iterable[tuple[int, int]] = (),  # (lyr, seq)
    mlp_head_locs: Iterable[tuple[int, int]] = (),  # (lyr, seq)
    embed_locs: Iterable[int] = (),  # seq
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
        for seq in embed_locs:
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
