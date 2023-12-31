import dataclasses
from typing import Callable

import einops
import numpy as np
import pandas
import torch
import transformer_lens.utils as tl_utils
from jaxtyping import Float
from tqdm.auto import tqdm
from transformer_lens import ActivationCache, HookedTransformer

from pii import ablation


@dataclasses.dataclass
class PromptData:
    prompt: str
    tokens: torch.Tensor
    token_strs: list[str]

    logits: torch.Tensor
    final_logits: torch.Tensor
    cache: ActivationCache

    # Maximum likelihood token, string, and probability
    ml_token: int
    ml_str: str
    ml_prob: float

    @property
    def n_tokens(self):
        return self.tokens.shape[0]

    @classmethod
    def get_data(cls, prompt: str, tl_model: HookedTransformer):
        with torch.no_grad():
            logits_b, cache = tl_model.run_with_cache(
                prompt,
                remove_batch_dim=True,
            )
            logits = tl_utils.remove_batch_dim(logits_b)
            final_logits = logits[-1]

        tokens = tl_model.to_tokens(prompt)[0]

        return cls(
            prompt=prompt,
            tokens=tokens,
            token_strs=[tl_model.to_string(t) for t in tokens],
            logits=logits,
            final_logits=logits[-1],
            cache=cache,
            ml_token=final_logits.argmax().item(),
            ml_str=tl_model.tokenizer.decode(final_logits.argmax()),
            ml_prob=final_logits.softmax(dim=-1).max().item(),
        )


@dataclasses.dataclass
class LogitData:
    pd: PromptData
    n_layers: int
    n_heads: int

    ablation_cache: ActivationCache | None

    attn_head_logits: Float[torch.Tensor, "layer head vocab"]
    attn_bias_logits: Float[torch.Tensor, "layer vocab"]
    mlp_logits: Float[torch.Tensor, "layer vocab"]
    embed_logits: Float[torch.Tensor, "vocab"]

    def get_labels(self, include_embedding: bool = False) -> list[str]:
        labels = []

        if include_embedding:
            labels.append("EMB")

        for lyr in range(self.n_layers):
            labels.append(f"L{lyr}ATB")
            labels.extend(f"L{lyr}H{h}ATN" for h in range(self.n_heads))
            labels.append(f"L{lyr}MLP")

        return labels

    def get_logits_and_labels(
        self,
        include_embedding: bool = False,
    ) -> tuple[Float[torch.Tensor, "idx vocab"], list[str]]:
        logits_list = []

        if include_embedding:
            logits_list.append(self.embed_logits[None, :])

        for lyr in range(self.n_layers):
            logits_list.append(self.attn_bias_logits[None, lyr, :])
            logits_list.append(self.attn_head_logits[lyr, :, :])
            logits_list.append(self.mlp_logits[None, lyr, :])

        return torch.concatenate(logits_list, dim=0), self.get_labels(
            include_embedding=include_embedding,
        )

    @classmethod
    def ablate_locs_dict(
        cls,
        labels: list[str],
    ) -> dict[str, list[tuple[int, ...]]]:
        """
        Converts labels into kwargs that can be fed into
        ablation.run_with_ablation.
        """
        return dict(
            attn_head_locs=[
                (
                    int(lab.split("H")[0][1:]),
                    int(lab.split("H")[1][:-3]),
                    -1,
                )
                for lab in labels
                if "ATN" in lab
            ],
            mlp_head_locs=[
                (int(lab.split("MLP")[0][1:]), -1)
                for lab in labels
                if "MLP" in lab
            ],
            attn_bias_locs=[
                (int(lab.split("ATB")[0][1:]), -1)
                for lab in labels
                if "ATB" in lab
            ],
            embed_locs=[(-1,)] if "EMB" in labels else [],
        )

    def get_cumulative_ablations(
        self,
        tl_model: HookedTransformer,
        target_token: int,
        top_k: int | None = 10,
        include_embedding: bool = False,
        increasing: bool = True,
    ) -> pandas.DataFrame:
        """
        Sorts components by log-odds contribution to the target_token, runs
        ablations on these components cumulatively, and returns both the true
        and predicted log-odds of the target token.

        If positive is True, then the components are sorted so that the ones
        that increase the log-odds of the target token when ablated come first.
        If positive is False, then the components are sorted so that the ones
        that decrease the log-odds of the target token when ablated come first.

        Logarithms are base e.
        """
        logits, labels = self.get_logits_and_labels(
            include_embedding=include_embedding,
        )
        top_k = top_k or len(labels)

        orig_log_odds = self.pd.final_logits.softmax(dim=-1).logit()[
            target_token
        ]
        log_odds = logits.softmax(dim=-1).logit()[:, target_token]
        log_bfs = log_odds - orig_log_odds  # log bayes factors

        components = sorted(
            [(log_bfs[i].item(), labels[i]) for i in range(len(labels))],
            reverse=increasing,
        )

        metrics = []
        for n_ablate in tqdm(range(top_k + 1)):
            labels = [label for _, label in components[:n_ablate]]
            log_bfs = [log_bf for log_bf, _ in components[:n_ablate]]

            final_logits = ablation.run_with_ablation(
                tl_model=tl_model,
                prompt=self.pd.prompt,
                return_type="logits",
                ablation_cache=self.ablation_cache,
                **self.ablate_locs_dict(labels),
            )[0, -1]

            logit = final_logits[target_token]
            prob = final_logits.softmax(dim=-1)[target_token]
            log_odds = prob.logit()

            metrics.append(
                dict(
                    n_ablate=n_ablate,
                    token=target_token,
                    token_str=tl_model.to_string(target_token),
                    label=labels[-1] if n_ablate > 0 else "None",
                    log_bf=log_bfs[-1] if n_ablate > 0 else 0,
                    logit=logit.item(),
                    prob=prob.item(),
                    log_odds=log_odds.item(),
                    pred_log_odds=orig_log_odds.item() + sum(log_bfs),
                )
            )

        df = pandas.DataFrame(metrics)
        assert np.isclose(
            df.logit[0], self.pd.final_logits[target_token].item()
        )

        return df


def get_ablated_logits(
    pd: PromptData,
    tl_model: HookedTransformer,
    ablation_cache: ActivationCache | None = None,
) -> LogitData:
    """
    Setting ablation_cache to None will do zero ablation.
    """

    def get_ablated_final_logits(**kwargs):
        return ablation.run_with_ablation(
            tl_model=tl_model,
            prompt=pd.prompt,
            return_type="logits",
            ablation_cache=ablation_cache,
            **kwargs,
        )[0, -1]

    embed_logits = get_ablated_final_logits(embed_locs=[(-1,)])

    mlp_logits = torch.stack(
        [
            get_ablated_final_logits(mlp_head_locs=[(layer, -1)])
            for layer in range(tl_model.cfg.n_layers)
        ]
    )

    attn_bias_logits = torch.stack(
        [
            get_ablated_final_logits(attn_bias_locs=[(layer, -1)])
            for layer in range(tl_model.cfg.n_layers)
        ]
    )

    attn_head_logits = torch.stack(
        [
            torch.stack(
                [
                    get_ablated_final_logits(attn_head_locs=[(l, h, -1)])
                    for h in range(tl_model.cfg.n_heads)
                ]
            )
            for l in tqdm(range(tl_model.cfg.n_layers))
        ]
    )

    return LogitData(
        pd=pd,
        n_layers=tl_model.cfg.n_layers,
        n_heads=tl_model.cfg.n_heads,
        ablation_cache=ablation_cache,
        embed_logits=embed_logits,
        mlp_logits=mlp_logits,
        attn_bias_logits=attn_bias_logits,
        attn_head_logits=attn_head_logits,
    )


def get_direct_effect_logits(
    tl_model: HookedTransformer,
    pd: PromptData,
    ablation_cache: ActivationCache | None = None,
) -> LogitData:
    """
    Ablates each component assuming additive independence.

    Setting ablation_cache to None will do zero ablation.
    """
    assert ablation_cache is None or (not ablation_cache.has_batch_dim)
    orig_resid: Float[torch.Tensor, "vocab"] = pd.cache["resid_post", -1][-1]

    def unembed(
        res: Float[torch.Tensor, "... d"]
    ) -> Float[torch.Tensor, "... vocab"]:
        return tl_model.ln_final(res) @ tl_model.W_U

    def _get_logits(
        fn: Callable[[ActivationCache], torch.Tensor]
    ) -> torch.Tensor:
        if ablation_cache is None:
            return unembed(orig_resid - fn(pd.cache))
        else:
            return unembed(orig_resid - fn(pd.cache) + fn(ablation_cache))

    embed_logits = _get_logits(lambda cache: cache["embed"][-1])

    mlp_logits = _get_logits(
        lambda cache: torch.stack(
            [cache["mlp_out", lyr][-1] for lyr in range(tl_model.cfg.n_layers)]
        )
    )

    attn_bias_logits = _get_logits(
        lambda _: torch.stack(
            [tl_model.b_O[lyr] for lyr in range(tl_model.cfg.n_layers)]
        )
    )

    if "blocks.0.attn.hook_result" not in pd.cache.cache_dict:
        pd.cache.compute_head_results()
    if (
        ablation_cache is not None
        and "blocks.0.attn.hook_result" not in ablation_cache.cache_dict
    ):
        ablation_cache.compute_head_results()
    attn_head_logits = _get_logits(
        lambda cache: torch.stack(
            [cache["result", lyr][-1] for lyr in range(tl_model.cfg.n_layers)]
        )
    )

    return LogitData(
        pd=pd,
        n_layers=tl_model.cfg.n_layers,
        n_heads=tl_model.cfg.n_heads,
        ablation_cache=ablation_cache,
        embed_logits=embed_logits,
        mlp_logits=mlp_logits,
        attn_bias_logits=attn_bias_logits,
        attn_head_logits=attn_head_logits,
    )


def attribution_patch(
    pd: PromptData,
    tl_model: HookedTransformer,
    metric: Callable[[torch.Tensor], torch.Tensor],
    ablation_cache: ActivationCache | None = None,
) -> LogitData:
    """
    Adapted from https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Attribution_Patching_Demo.ipynb

    metric is a function that takes logits and returns a scalar.
    Slightly abuses LogitData, since it returns logit data with |vocab| = 1.
    """
    tl_model.reset_hooks()
    _grad_cache = {}

    def backward_cache_hook(act, hook):
        _grad_cache[hook.name] = act[0].detach()

    backward_hook_locs = set(
        [tl_utils.get_act_name("embed")]
        + [
            tl_utils.get_act_name("mlp_out", lyr)
            for lyr in range(tl_model.cfg.n_layers)
        ]
        + [
            tl_utils.get_act_name("resid_mid", lyr)
            for lyr in range(tl_model.cfg.n_layers)
        ]
        + [
            tl_utils.get_act_name("z", lyr)
            for lyr in range(tl_model.cfg.n_layers)
        ]
    )
    tl_model.add_hook(
        lambda name: name in backward_hook_locs,
        backward_cache_hook,
        "bwd",
    )

    value = metric(tl_model(pd.tokens)[0])
    value.backward()
    tl_model.reset_hooks()

    gcache = ActivationCache(_grad_cache, tl_model)

    def linearized_ablation(
        get_act: Callable[[ActivationCache], torch.Tensor],
        get_grad: Callable[[ActivationCache], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if get_grad is None:
            get_grad = get_act

        if ablation_cache is None:
            # Zero ablation
            return -einops.einsum(
                get_grad(gcache),
                get_act(pd.cache),
                "... d , ... d -> ...",
            )
        else:
            return einops.einsum(
                get_grad(gcache),
                get_act(ablation_cache) - get_act(pd.cache),
                "... d , ... d -> ...",
            )

    embed_metrics = linearized_ablation(lambda cache: cache["embed"][-1])

    mlp_metrics = linearized_ablation(
        lambda cache: torch.stack(
            [cache["mlp_out", lyr][-1] for lyr in range(tl_model.cfg.n_layers)]
        )
    )

    attn_bias_metrics = linearized_ablation(
        get_act=lambda _: torch.stack(
            [tl_model.b_O[lyr] for lyr in range(tl_model.cfg.n_layers)]
        ),
        get_grad=lambda gcache: torch.stack(
            [
                gcache["resid_mid", lyr][-1]
                for lyr in range(tl_model.cfg.n_layers)
            ]
        ),
    )

    attn_head_metrics = linearized_ablation(
        lambda cache: torch.stack(
            [cache["z", lyr][-1] for lyr in range(tl_model.cfg.n_layers)]
        )
    )

    return LogitData(
        pd=pd,
        n_layers=tl_model.cfg.n_layers,
        n_heads=tl_model.cfg.n_heads,
        ablation_cache=ablation_cache,
        embed_logits=embed_metrics.unsqueeze(-1),
        mlp_logits=mlp_metrics.unsqueeze(-1),
        attn_bias_logits=attn_bias_metrics.unsqueeze(-1),
        attn_head_logits=attn_head_metrics.unsqueeze(-1),
    )
