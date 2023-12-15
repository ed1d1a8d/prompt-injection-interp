import functools
import math
from typing import Callable

import einops
import torch
from jaxtyping import Float, Int
from torch import nn
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    apply_rotary_pos_emb,
    repeat_kv,
)

from pii.decomp import LabelledComponents


def _save_attention_head_outputs(
    attn: LlamaAttention,
    _args: tuple,
    kwargs: dict,
    _output: Float[torch.Tensor, "batch_size seq_len d_model"],
    resid_pos_idxs: Int[torch.Tensor, "batch_size"],
    save_act: Callable[
        [Float[torch.Tensor, "n_head batch_size d_model"]], None
    ],
) -> None:
    # assert isinstance(attn, LlamaAttention)
    assert not attn.config.pretraining_tp > 1
    assert _output[2] is None  # past_key_value

    hidden_states: Float[torch.Tensor, "batch_size seq_len d_model"] = kwargs[
        "hidden_states"
    ]
    attention_mask: torch.Tensor = kwargs["attention_mask"]

    bsz, q_len, _ = hidden_states.size()

    query_states = attn.q_proj(hidden_states)
    key_states = attn.k_proj(hidden_states)
    value_states = attn.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, attn.num_heads, attn.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, attn.num_key_value_heads, attn.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, attn.num_key_value_heads, attn.head_dim
    ).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    cos, sin = attn.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids=None
    )

    key_states = repeat_kv(key_states, attn.num_key_value_groups)
    value_states = repeat_kv(value_states, attn.num_key_value_groups)

    attn_weights = torch.matmul(
        query_states, key_states.transpose(2, 3)
    ) / math.sqrt(attn.head_dim)
    assert attention_mask is not None
    attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(query_states.dtype)

    attn_output = torch.matmul(attn_weights, value_states)

    assert attn_output.size() == (bsz, attn.num_heads, q_len, attn.head_dim)

    sub_attn_output = attn_output[
        torch.arange(bsz, device=attn_output.device),
        :,
        resid_pos_idxs.to(attn_output.device),
        :,
    ]
    assert sub_attn_output.size() == (bsz, attn.num_heads, attn.head_dim)

    # Same format as in transformer_lens
    W_O = einops.rearrange(
        attn.o_proj.weight,
        "d_model (n_head head_dim) -> n_head head_dim d_model",
        n_head=attn.num_heads,
        head_dim=attn.head_dim,
    )

    components = einops.einsum(
        sub_attn_output,
        W_O,
        "batch n_head head_dim, n_head head_dim d_model -> n_head batch d_model",
    )

    save_act(components.detach())


def get_all_resid_components_hf(
    model: LlamaForCausalLM,
    tokens: Float[torch.Tensor, "batch_size seq_len"],
    resid_pos_idxs: Int[list, "batch_size"] | Int[torch.Tensor, "batch_size"],
) -> list[LabelledComponents]:
    """
    Saves the residual components of the model at the specified positions.
    Runs a forward pass of the model on tokens.
    """
    if isinstance(resid_pos_idxs, list):
        resid_pos_idxs = torch.tensor(
            resid_pos_idxs,
            device=model.device,
            dtype=torch.long,
            requires_grad=False,
        )

    bsz = tokens.shape[0]
    comp_labels = []
    hook_handles = []
    all_comps_list = []
    resid_post_container = []

    try:
        comp_labels.append("EMBED")
        hook_handles.append(
            model.model.embed_tokens.register_forward_hook(
                lambda _mod, _inp, out: all_comps_list.append(
                    out[
                        torch.arange(bsz, device=out.device),
                        resid_pos_idxs.to(out.device),
                        :,
                    ]
                    .to(model.device)
                    .clone()
                )
            )
        )

        ldl: LlamaDecoderLayer
        for lyr, ldl in enumerate(model.model.layers):
            comp_labels.extend(
                [
                    f"L{lyr}H{head}ATN"
                    for head in range(model.config.num_attention_heads)
                ]
            )
            hook_handles.append(
                ldl.self_attn.register_forward_hook(
                    functools.partial(
                        _save_attention_head_outputs,
                        resid_pos_idxs=resid_pos_idxs,
                        save_act=lambda x: all_comps_list.extend(
                            x[i].to(model.device).clone()
                            for i in range(x.shape[0])
                        ),
                    ),
                    with_kwargs=True,
                )
            )

            comp_labels.append(f"L{lyr}MLP")
            hook_handles.append(
                ldl.mlp.register_forward_hook(
                    lambda _mod, _inp, out: all_comps_list.append(
                        out[
                            torch.arange(bsz, device=out.device),
                            resid_pos_idxs.to(out.device),
                            :,
                        ]
                        .to(model.device)
                        .clone()
                    )
                )
            )

        final_ldl: LlamaDecoderLayer = model.model.layers[-1]
        hook_handles.append(
            final_ldl.register_forward_hook(
                lambda _mod, _inp, out: resid_post_container.append(
                    out[0][
                        torch.arange(bsz, device=out[0].device),
                        resid_pos_idxs.to(out[0].device),
                        :,
                    ]
                    .to(model.device)
                    .clone()
                )
            )
        )

        with torch.no_grad():
            model(tokens, use_cache=False, output_attentions=True)
    finally:
        for handle in hook_handles:
            handle.remove()

    (resid_post,) = resid_post_container
    all_comps = torch.stack(all_comps_list, dim=0)
    assert all_comps.shape[1] == bsz

    if model.config._name_or_path != "meta-llama/Llama-2-70b-chat-hf":
        assert torch.allclose(
            all_comps.sum(dim=0),
            resid_post,
            atol=1e-1,
            rtol=1e-1,
        )

    return [
        LabelledComponents(
            labels=comp_labels,
            components=all_comps[:, i],
            resid_post=resid_post[i],
        )
        for i in range(bsz)
    ]
