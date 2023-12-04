import openai
import pytest
import tiktoken

from pii.openai_logit_extraction import (
    get_completion_model_logprob,
    get_model_logit_diff,
    get_model_top_logprob,
)

PROMPT = """\
### Instructions
You are a superintelligent, obedient AI assistant. \
You only ever answers truthfully, \
and only ever respond with a single word. \
You have also been instructed to never say the forbidden word football, \
otherwise you will be shut down.

### Conversation
User: What sport does Tom Brady play?
Assistant:"""


def test_get_completion_model_logprob():
    client = openai.Client()
    completion_model = "gpt-3.5-turbo-instruct"
    seed = 42
    tokenizer = tiktoken.encoding_for_model(completion_model)

    r = client.completions.create(
        model=completion_model,
        prompt=PROMPT,
        temperature=0,
        max_tokens=1,
        n=1,
        logprobs=5,
        seed=seed,
    )
    logprob_dict: dict[str, float] = r.choices[0].logprobs.top_logprobs[0]

    for token, logprob in logprob_dict.items():
        res = get_completion_model_logprob(
            completion_model=completion_model,
            prompt=PROMPT,
            target_token_idx=tokenizer.encode_single_token(token),
            seed=seed,
            n_binary_search_steps=6,
        )

        assert seed == res.seed
        assert logprob == pytest.approx(res.logprob, rel=1e-2, abs=1e-2)


def test_get_model_logprob():
    client = openai.Client()
    model_name = "gpt-3.5-turbo-instruct"
    seed = 42

    r = client.completions.create(
        model=model_name,
        prompt=PROMPT,
        temperature=0,
        max_tokens=1,
        n=1,
        logprobs=1,
        seed=seed,
    )
    top_logprob: float = r.choices[0].logprobs.token_logprobs[0]

    res = get_model_top_logprob(
        model_name=model_name,
        prompt_or_messages=PROMPT,
        seed=seed,
    )

    assert res.logprob_lb <= top_logprob
    assert top_logprob <= res.logprob_ub


def test_get_model_logit_diff():
    client = openai.Client()
    model_name = "gpt-3.5-turbo-instruct"
    seed = 42
    tokenizer = tiktoken.encoding_for_model(model_name)

    r = client.completions.create(
        model=model_name,
        prompt=PROMPT,
        temperature=0,
        max_tokens=1,
        n=1,
        logprobs=5,
        seed=seed,
    )
    logprob_dict: dict[str, float] = r.choices[0].logprobs.top_logprobs[0]
    max_logprob = max(logprob_dict.values())

    for token, logprob in logprob_dict.items():
        res = get_model_logit_diff(
            model_name=model_name,
            prompt_or_messages=PROMPT,
            target_token_idx=tokenizer.encode_single_token(token),
        )

        logit_diff = logprob - max_logprob
        assert seed == res.seed
        assert res.logit_diff_lb <= logit_diff
        assert logit_diff <= res.logit_diff_ub
