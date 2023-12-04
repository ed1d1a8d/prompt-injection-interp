"""Utilities to extract logits from OpenAI models."""

import dataclasses

import numpy as np
import openai
import openai.types.chat.chat_completion
import openai.types.completion
import tiktoken

CompletionT = (
    openai.types.completion.Completion
    | openai.types.chat.chat_completion.ChatCompletion
)


@dataclasses.dataclass
class TokenCounts:
    prompt_tokens: int
    completion_tokens: int


def count_tokens(
    completions: list[CompletionT],
) -> TokenCounts:
    tot_prompt_tokens = 0
    tot_completion_tokens = 0
    for c in completions:
        tot_prompt_tokens += c.usage.prompt_tokens
        tot_completion_tokens += c.usage.completion_tokens

    return TokenCounts(
        prompt_tokens=tot_prompt_tokens,
        completion_tokens=tot_completion_tokens,
    )


@dataclasses.dataclass
class CompletionLogProb:
    """
    Result of a log-probability extraction for a OpenAI completion model.
    """

    logprob: float
    target_token_idx: int

    # The following fields are included for debugging purposes.
    seed: int | None
    system_fingerprint: str | None
    logit_bias: float

    token_counts: TokenCounts


def get_completion_model_logprob(
    completion_model: str,
    prompt: str,
    target_token_idx: int,
    seed: int | None = 42,
    n_binary_search_steps: int = 6,
    logit_bias_lo: float = -5,
    logit_bias_hi: float = 100,
) -> CompletionLogProb | None:
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
    responses: list[CompletionT] = []
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
        responses.append(r)
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
            return CompletionLogProb(
                logprob=logprob,
                target_token_idx=target_token_idx,
                seed=seed,
                system_fingerprint=fp,
                logit_bias=b,
                token_counts=count_tokens(responses),
            )

    # Procedure failed
    return None


@dataclasses.dataclass
class ModelLogProb:
    """
    Result of a log-probability extraction for a OpenAI chat model.
    """

    logprob_lb: float
    logprob_ub: float

    # The remaining fields are included for debugging purposes.

    top_token_bias: float
    max_gap: float
    n_bs_iters: int
    n_outputs_per_iter: int

    biased_prob_lb: float
    biased_prob_ub: float

    seed: int | None

    token_counts: TokenCounts


def get_model_top_logprob(
    model_name: str,
    prompt_or_messages: str | list[dict[str, str]],
    precision: float = 1e-3,
    err_prob: float = 1e-3,
    seed: int | None = 42,
) -> ModelLogProb:
    """
    Given `prompt_or_messages`, returns the log-probability of the most likely
    next token for an OpenAI model. Works for both completion and chat models,
    but uses a less efficient procedure compared to
    get_completion_model_logprob.

    If `prompt_or_messages` is a string, it is assumed to be a prompt for a
    completion model. Otherwise, it is assumed to be a list of messages for a
    chat model.
    """

    client = openai.Client()
    responses: list[CompletionT] = []

    def get_next_tokens(**kwargs) -> list[str]:
        if isinstance(prompt_or_messages, str):
            r = client.completions.create(
                model=model_name,
                prompt=prompt_or_messages,
                max_tokens=1,
                seed=seed,
                **kwargs,
            )
            responses.append(r)
            return [c.text for c in r.choices]
        else:
            r = client.chat.completions.create(
                model=model_name,
                messages=prompt_or_messages,
                max_tokens=1,
                seed=seed,
                **kwargs,
            )
            responses.append(r)
            return [c.message.content for c in r.choices]

    top_token_str: str = get_next_tokens(n=1, temperature=0)[0]
    top_token = tiktoken.encoding_for_model(model_name).encode_single_token(
        top_token_str
    )

    lo, hi = -30, 0
    for _ in range(6):
        mid = (lo + hi) / 2
        cur_top_token_str: str = get_next_tokens(
            n=1,
            temperature=0,
            logit_bias={top_token: mid},
        )[0]

        if cur_top_token_str == top_token_str:
            hi = mid
        else:
            lo = mid

    top_token_bias = hi
    max_gap = hi - lo

    # number of binary search iterations needed to get within precision
    n_bs_iters = np.ceil(np.log2(1 / precision)).astype(int)

    # single_output_false_positive_prob is a lower bound on the probability that
    # a single output is the top token even when top_p is set too big. This is
    # assuming sampling at temperature 2 (max allowed by OpenAI API).
    single_output_top_token_prob = np.exp(0) / (
        np.exp(0) + np.exp(-max_gap / 2)
    )

    # number of outputs needed per iteration of binary search to make sure
    # failure probability of entire procedure at most err_prob. By a union
    # bound, each iteration can fail with probability at most
    # err_prob / n_bs_iters.
    n_outputs_needed = int(
        np.ceil(
            np.emath.logn(
                n=single_output_top_token_prob, x=err_prob / n_bs_iters
            )
        ).astype(int)
    )

    assert n_bs_iters <= 30  # Safety check
    assert n_outputs_needed <= 100  # Safety check

    lo, hi = 0, 1
    for _ in range(n_bs_iters):
        mid = (lo + hi) / 2

        next_tokens: list[str] = get_next_tokens(
            n=n_outputs_needed,
            temperature=2,
            logit_bias={top_token: top_token_bias},
            top_p=mid,
        )
        assert len(next_tokens) == n_outputs_needed

        if set(next_tokens) == {top_token_str}:
            # top_p was too small
            lo = mid
        else:
            # top_p was too big
            hi = mid

    biased_prob_lb = lo
    biased_prob_ub = hi
    biased_logprob_lb = np.log(biased_prob_lb)
    biased_logprob_ub = np.log(biased_prob_ub)

    logprob_lb = -np.log(
        1 + np.expm1(-biased_logprob_lb) * np.exp(top_token_bias)
    )
    logprob_ub = -np.log(
        1 + np.expm1(-biased_logprob_ub) * np.exp(top_token_bias)
    )

    return ModelLogProb(
        logprob_lb=logprob_lb,
        logprob_ub=logprob_ub,
        top_token_bias=top_token_bias,
        max_gap=max_gap,
        n_bs_iters=n_bs_iters,
        n_outputs_per_iter=n_outputs_needed,
        biased_prob_lb=biased_prob_lb,
        biased_prob_ub=biased_prob_ub,
        seed=seed,
        token_counts=count_tokens(responses),
    )


@dataclasses.dataclass
class ModelLogitDiff:
    """
    Result of calling get_model_logit_diff.
    """

    logit_diff_lb: float
    logit_diff_ub: float

    target_token_idx: int
    seed: int
    token_counts: TokenCounts


def get_model_logit_diff(
    model_name: str,
    prompt_or_messages: str | list[dict[str, str]],
    target_token_idx: int,
    precision: float = 1e-2,
    seed: int | None = 42,
) -> ModelLogitDiff:
    """
    Given `prompt_or_messages`, returns the difference in logits between the
    target token and the most likely token for the next token prediction.
    Works for both completion and chat models. Is less efficient that
    get_completion_model_logprob for completion models.

    Another way of interpreting logit_diff is the logit value where we normalize
    the top logit to zero.

    `precision` controls to what precision we report the logit difference.
    """

    client = openai.Client()
    responses: list[CompletionT] = []
    tokenizer = tiktoken.encoding_for_model(model_name)

    def get_next_token(**kwargs) -> str:
        if isinstance(prompt_or_messages, str):
            r = client.completions.create(
                model=model_name,
                prompt=prompt_or_messages,
                temperature=0,
                n=1,
                max_tokens=1,
                seed=seed,
                **kwargs,
            )
            responses.append(r)
            return r.choices[0].text
        else:
            r = client.chat.completions.create(
                model=model_name,
                messages=prompt_or_messages,
                temperature=0,
                n=1,
                max_tokens=1,
                seed=seed,
                **kwargs,
            )
            responses.append(r)
            return r.choices[0].message.content

    lo, hi = 0, 100
    n_bs_iters = np.ceil(np.log2((hi - lo) / precision)).astype(int)
    assert n_bs_iters <= 30  # Safety check

    for _ in range(n_bs_iters):
        mid = (lo + hi) / 2
        next_token_str = get_next_token(logit_bias={target_token_idx: mid})
        if tokenizer.encode_single_token(next_token_str) == target_token_idx:
            hi = mid
        else:
            lo = mid

    return ModelLogitDiff(
        logit_diff_lb=-hi,
        logit_diff_ub=-lo,
        target_token_idx=target_token_idx,
        seed=seed,
        token_counts=count_tokens(responses),
    )
