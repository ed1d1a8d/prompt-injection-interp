import torch
import transformer_lens.utils as tl_utils
from transformer_lens import HookedTransformer


def get_top_responses(
    prompt: str,
    model: HookedTransformer,
    top_k: int = 5,
    n_continuation_tokens: int = 5,
    prepend_bos: bool | None = None,
    print_prompt: bool = False,
) -> None:
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
        )[0][prompt_tokens.shape[1] :]

        print(
            f"Rank {i}. "
            f"Logit: {logit:5.2f} "
            f"Prob: {prob:6.2%} "
            f"Tokens: |{'|'.join([model.to_string(t) for t in continuation_tokens])}|"
        )
