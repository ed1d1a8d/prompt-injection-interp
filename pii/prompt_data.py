import dataclasses

import einops
import torch
import transformer_lens.utils as tl_utils
from jaxtyping import Float
from transformer_lens import ActivationCache, HookedTransformer
from tuned_lens.nn import LogitLens, TunedLens
from tuned_lens.plotting import PredictionTrajectory


@dataclasses.dataclass
class PromptData:
    prompt: str
    tokens: torch.Tensor
    token_strs: list[str]

    logits: torch.Tensor
    final_logits: torch.Tensor
    cache: ActivationCache

    # Maximum likelihood token and string
    ml_token: int
    ml_str: str

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
        )


@dataclasses.dataclass
class LensData:
    pd: PromptData
    n_layers: int
    n_heads: int

    # Only stores last token logits
    logits_dict: dict[
        str,
        Float[torch.Tensor, "layer head vocab"]
        | Float[torch.Tensor, "layer vocab"],
    ]

    @classmethod
    def get_key(cls, residual_component: str, lens_type: str):
        return f"{residual_component}_{lens_type}"

    def get_logits(
        self,
        residual_component: str,
        lens_type: str,
        zero_mean: bool = True,
        flatten: bool = True,
    ) -> tuple[torch.Tensor, list[str]]:
        logits = self.logits_dict[
            self.get_key(residual_component, lens_type)
        ].clone()

        labels = (
            [
                f"L{layer}H{head}"
                for layer in range(self.n_layers)
                for head in range(self.n_heads)
            ]
            if logits.ndim == 3
            else [f"L{layer}" for layer in range(self.n_layers)]
        )

        if zero_mean:
            logits -= logits.mean(dim=-1, keepdim=True)

        if flatten and logits.ndim == 3:
            logits = einops.rearrange(
                logits, "layer head vocab -> (layer head) vocab"
            )

        return logits, labels

    @classmethod
    def get_data(
        cls,
        prompt_data: PromptData,
        tl_model: HookedTransformer,
        logit_lens: LogitLens,
        tuned_lens: TunedLens,
        res_components: tuple[str, ...] = (
            "resid_pre",
            "attn_out",
            "mlp_out",
        ),
    ):
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
