import dataclasses
import functools

import einops
import numpy as np
import torch
import transformer_lens.utils as tl_utils
from jaxtyping import Float
from tqdm.auto import tqdm
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.hook_points import HookPoint
from tuned_lens.nn import LogitLens, TunedLens


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
class LogitData:
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

    @classmethod
    def get_resid_code(cls, residual_component: str) -> str:
        return dict(
            resid_pre="PRE",
            attn_out="ATN",
            attn_head="ATH",
            mlp_out="MLP",
        ).get(residual_component, residual_component)

    def get_logits(
        self,
        residual_component: str,
        lens_type: str,
        zero_mean: bool = True,
        flatten: bool = True,
    ) -> tuple[torch.Tensor, np.ndarray]:
        logits = self.logits_dict[
            self.get_key(residual_component, lens_type)
        ].clone()

        resid_code = self.get_resid_code(residual_component)
        labels = (
            np.array(
                [
                    [
                        f"L{layer}H{head}{resid_code}"
                        for head in range(self.n_heads)
                    ]
                    for layer in range(self.n_layers)
                ]
            )
            if logits.ndim == 3
            else np.array(
                [f"L{layer}{resid_code}" for layer in range(self.n_layers)]
            )
        )

        if zero_mean:
            logits -= logits.mean(dim=-1, keepdim=True)

        if flatten and logits.ndim == 3:
            logits = einops.rearrange(
                logits, "layer head vocab -> (layer head) vocab"
            )
            labels = labels.flatten()

        return logits, labels

    @classmethod
    def get_lens_data(
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

    @classmethod
    def get_patch_data(
        cls,
        pd: PromptData,
        tl_model: HookedTransformer,
        opd: PromptData | None = None,
        res_components: tuple[str, ...] = (
            "attn_out",
            "mlp_out",
        ),
    ):
        """
        Setting opd to None will do zero ablation.
        """
        logits_dict = dict()

        def patch_res(
            act: Float[torch.Tensor, "batch seq d"],
            hook: HookPoint,
            layer: int,
            res_str: str,
        ) -> Float[torch.Tensor, "batch seq head d"]:
            act[:, -1] = (
                0 if opd is None else opd.cache[res_str, layer][None, -1]
            )
            return act

        for res_str in res_components:
            logits_dict[cls.get_key(res_str, "patch")] = torch.stack(
                [
                    tl_model.run_with_hooks(
                        pd.prompt,
                        return_type="logits",
                        fwd_hooks=[
                            (
                                tl_utils.get_act_name(res_str, layer),
                                functools.partial(
                                    patch_res, layer=layer, res_str=res_str
                                ),
                            )
                        ],
                    )[0, -1]
                    for layer in range(tl_model.cfg.n_layers)
                ]
            )

        def patch_attn_head(
            zs: Float[torch.Tensor, "batch seq head d"],
            hook: HookPoint,
            head: int,
            layer: int,
        ) -> Float[torch.Tensor, "batch seq head d"]:
            zs[:, -1, head] = (
                0 if opd is None else opd.cache["z", layer][None, -1, head]
            )
            return zs

        lhs = [
            (l, h)
            for l in range(tl_model.cfg.n_layers)
            for h in range(tl_model.cfg.n_heads)
        ]
        attn_logits = torch.zeros(
            (len(lhs), tl_model.cfg.d_vocab), device=pd.cache["z", 0].device
        )
        for i, (l, h) in enumerate(tqdm(lhs)):
            attn_logits[i] = tl_model.run_with_hooks(
                pd.prompt,
                return_type="logits",
                fwd_hooks=[
                    (
                        tl_utils.get_act_name("z", l),
                        functools.partial(patch_attn_head, head=h, layer=l),
                    )
                ],
            )[0, -1]

        logits_dict[cls.get_key("attn_head", "patch")] = einops.rearrange(
            attn_logits,
            "(layer head) vocab -> layer head vocab",
            layer=tl_model.cfg.n_layers,
        )

        return cls(
            pd=pd,
            n_layers=tl_model.cfg.n_layers,
            n_heads=tl_model.cfg.n_heads,
            logits_dict=logits_dict,
        )
