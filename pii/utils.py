import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import transformer_lens.utils as tl_utils
from transformer_lens import HookedTransformer


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
    lines: list[tuple[str, torch.Tensor | np.ndarray | pd.Series, list[str]]],
    annotation_text: str | None = None,
    spacing: int = 4 * 32,
    **kwargs,
):
    fig = go.Figure()
    for name, xs, labels in lines:
        if isinstance(xs, torch.Tensor):
            xs = xs.flatten().cpu().numpy()
        elif isinstance(xs, pd.Series):
            xs = xs.values
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
    fig.update_layout(
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
