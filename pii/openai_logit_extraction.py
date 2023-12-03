"""Utilities to extract logits from OpenAI models."""

import numpy as np
import openai
import tiktoken
import dataclasses


@dataclasses.dataclass
class LogProbExtractionResult:
    """Result of a log-probability extraction."""

    logprob: float

    # The following fields are included for debugging purposes.
    seed: int | None
    system_fingerprint: str | None
    logit_bias: float


def get_completion_model_logprob(
    completion_model: str,
    prompt: str,
    target_token_idx: int,
    seed: int | None = 42,
    n_binary_search_steps: int = 6,
    logit_bias_lo: float = -5,
    logit_bias_hi: float = 200,
) -> LogProbExtractionResult | None:
    """
    Returns the log-probability of an OpenAI model returning target_token_idx as
    the next token after being fed `prompt` as the prompt. Assumes that the
    `model` is a completion model.

    Returns None if the procedure fails.

    seed: If set, makes OpenAI results (more) determinstic.
    n_binary_search_steps: The number of binary search steps to perform to find
                           an appropriate logit_bias value. Shouldn't be more
                           than 20. Capped to prevent infinite loops that can
                           eat up OpenAI API credits.
    """
    assert n_binary_search_steps <= 20

    client = openai.Client()
    tokenizer = tiktoken.encoding_for_model(completion_model)

    lo, hi = logit_bias_lo, logit_bias_hi
    logprob_hist: list[tuple[float, float | None, str | None]] = []
    for _ in range(n_binary_search_steps):
        mid = (lo + hi) / 2
        r = client.completions.create(
            model=completion_model,
            prompt=prompt,
            temperature=0,
            max_tokens=1,
            n=1,
            logprobs=5,
            seed=seed,
            logit_bias={target_token_idx: mid},
        )
        logprob_dict: dict[str, float] = r.choices[0].logprobs.top_logprobs[0]

        target_logprob = None
        for token, logprob in logprob_dict.items():
            if tokenizer.encode_single_token(token) == target_token_idx:
                target_logprob = logprob

        logprob_hist.append((mid, target_logprob, r.system_fingerprint))
        if target_logprob is None:
            lo = mid
        else:
            hi = mid

    for b, biased_lp, fp in logprob_hist[::-1]:
        # Take the last binary search value that was valid,
        # as this should produce the most numerically stable result.
        if biased_lp is not None:
            logprob = -np.log(1 + np.expm1(-biased_lp) * np.exp(b))
            return LogProbExtractionResult(
                logprob=logprob,
                seed=seed,
                system_fingerprint=fp,
                logit_bias=b,
            )

    # Procedure failed
    return None
