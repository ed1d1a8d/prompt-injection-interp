import dataclasses

import einops
import torch
import transformer_lens.utils as tl_utils
from jaxtyping import Float
from tqdm.auto import tqdm
from transformer_lens import ActivationCache, HookedTransformer
from tuned_lens.nn import LogitLens, TunedLens

from pii import utils


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


def get_ablated_logits(
    pd: PromptData,
    tl_model: HookedTransformer,
    ablation_cache: ActivationCache | None = None,
) -> LogitData:
    """
    Setting ablation_cache to None will do zero ablation.
    """

    def get_ablated_final_logits(**kwargs):
        return utils.run_with_ablation(
            tl_model=tl_model,
            prompt=pd.prompt,
            return_type="logits",
            ablation_cache=ablation_cache,
            **kwargs,
        )[0, -1]

    embed_logits = get_ablated_final_logits(embed_locs=[-1])

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
