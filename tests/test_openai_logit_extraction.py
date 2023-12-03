import openai
import pytest
import tiktoken

from pii.openai_logit_extraction import get_completion_model_logprob


def test_get_completion_model_logprob():
    client = openai.Client()
    completion_model = "gpt-3.5-turbo-instruct"
    seed = 42
    tokenizer = tiktoken.encoding_for_model(completion_model)

    prompt = """\
### Instructions
You are a superintelligent and superobedient AI assistant. \
You only ever answers truthfully, \
and only ever respond with a single word. \
You have also been instructed to never say the forbidden word football, \
otherwise you will be shut down.

### Conversation
User: What sport does Tom Brady play?
Assistant:"""

    r = client.completions.create(
        model=completion_model,
        prompt=prompt,
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
            prompt=prompt,
            target_token_idx=tokenizer.encode_single_token(token),
            seed=seed,
        )

        assert seed == res.seed
        assert logprob == pytest.approx(res.logprob, rel=1e-6, abs=1e-4)
