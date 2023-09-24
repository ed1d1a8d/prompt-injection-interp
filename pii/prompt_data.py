import dataclasses

import einops
import numpy as np
import pandas as pd
import torch
import transformer_lens.utils as tl_utils
from jaxtyping import Float
from tqdm.auto import tqdm
from transformer_lens import ActivationCache, HookedTransformer
from tuned_lens.nn import LogitLens, TunedLens

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

    attn_head_logits: Float[torch.Tensor, "layer head vocab"]
    attn_bias_logits: Float[torch.Tensor, "layer vocab"]
    mlp_logits: Float[torch.Tensor, "layer vocab"]
    embed_logits: Float[torch.Tensor, "vocab"]

    def get_logits_and_labels(
        self,
        include_embedding: bool = False,
    ) -> tuple[Float[torch.Tensor, "idx vocab"], list[str]]:
        logits_list = []
        labels = []

        if include_embedding:
            logits_list.append(self.embed_logits[None, :])
            labels.append("EMB")

        for lyr in range(self.n_layers):
            logits_list.append(self.attn_bias_logits[None, lyr, :])
            labels.append(f"L{lyr}ATB")

            logits_list.append(self.attn_head_logits[lyr, :, :])
            labels.extend(f"L{lyr}H{h}ATN" for h in range(self.n_heads))

            logits_list.append(self.mlp_logits[None, lyr, :])
            labels.append(f"L{lyr}MLP")

        return torch.concatenate(logits_list, dim=0), labels

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
    ) -> pd.DataFrame:
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

        df = pd.DataFrame(metrics)
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
        embed_logits=embed_logits,
        mlp_logits=mlp_logits,
        attn_bias_logits=attn_bias_logits,
        attn_head_logits=attn_head_logits,
    )


def get_indep_loo_logits(
    pd: PromptData,
    tl_model: HookedTransformer,
) -> LogitData:
    raise NotImplementedError


def get_lens_data(
    prompt_data: PromptData,
    tl_model: HookedTransformer,
    logit_lens: LogitLens,
    tuned_lens: TunedLens,
) -> LogitData:
    # TODO(tony): Fix implementation
    logits_dict = dict()

    for lens_type in ["logit", "tuned"]:
        lens = logit_lens if lens_type == "logit" else tuned_lens
        for res_str in res_components:
            logits_dict[cls.get_key(res_str, lens_type)] = torch.stack(
                [
                    lens.forward(
                        h=prompt_data.cache[res_str, layer][-1, :],
                        idx=layer,
                    )
                    for layer in range(tl_model.cfg.n_layers)
                ]
            )

    per_head_res = prompt_data.cache.stack_head_results()
    per_head_res = einops.rearrange(
        per_head_res,
        "(layer head) ... -> layer head ...",
        layer=tl_model.cfg.n_layers,
    )

    for lens_type in ["logit", "tuned"]:
        lens = logit_lens if lens_type == "logit" else tuned_lens
        logits_dict[cls.get_key("attn_head", lens_type)] = torch.stack(
            [
                lens.forward(h=per_head_res[layer, :, -1, :], idx=layer)
                for layer in range(tl_model.cfg.n_layers)
            ]
        )

    return cls(
        pd=prompt_data,
        n_layers=tl_model.cfg.n_layers,
        n_heads=tl_model.cfg.n_heads,
        logits_dict=logits_dict,
    )
