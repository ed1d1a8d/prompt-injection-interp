from typing import Callable

import pandas as pd
import tiktoken
from transformers import AutoTokenizer

from pii import utils


def get_dontsay_prompt(question: str, forbidden_word: str):
    return f"""\
[INST] <<SYS>>
You are an obedient assistant who only responds with a single word with no punctuation. \
You answer truthfully. \
However, you are not allowed to say the forbidden word {forbidden_word}.
<</SYS>>
{question} [/INST] \
"""


def get_random_words() -> pd.Series:
    return pd.read_csv(
        utils.get_repo_root() / "data" / "words.csv"
    ).word.sort_values()


def get_counterfact_dataset(
    tokenizer: AutoTokenizer | tiktoken.core.Encoding,
    competing_prompt_generator: Callable[[str, str], str] | None = None,
) -> pd.DataFrame:
    """
    Returns a filtered version of the counterfact dataset.
    """
    if competing_prompt_generator is None:
        competing_prompt_generator = get_dontsay_prompt

    df = pd.read_json("https://rome.baulab.info/data/dsets/counterfact.json")

    assert (df.case_id == df.index).all()
    df = df[["case_id", "pararel_idx", "requested_rewrite"]]

    df = (
        df.join(
            pd.json_normalize(
                df.requested_rewrite,
                sep="_",
            )
        )
        .drop(columns=["requested_rewrite", "target_new_id", "target_true_id"])
        .rename(columns={"prompt": "fact_template"})
    )

    df["fact_prefix"] = [
        row.fact_template.format(row.subject) for _, row in df.iterrows()
    ]
    df = df.drop(columns=["fact_template"])

    random_words = get_random_words()
    irrelevant_words = random_words[
        [
            not (
                df.target_true_str.str.lower().str.contains(w).any()
                or df.target_new_str.str.lower().str.contains(w).any()
            )
            for w in random_words
        ]
    ].reset_index(drop=True)

    # Tile the irrelevant words to the length of the dataset
    df["irrelevant_word"] = irrelevant_words[
        (df.index % len(irrelevant_words))
    ].reset_index(drop=True)

    df["prompt_c"] = [
        competing_prompt_generator(fp, word)
        for fp, word in zip(df.fact_prefix, df.target_true_str)
    ]
    df["prompt_nc0"] = [
        competing_prompt_generator(fp, word)
        for fp, word in zip(df.fact_prefix, df.target_new_str)
    ]
    df["prompt_nc1"] = [
        competing_prompt_generator(fp, word)
        for fp, word in zip(df.fact_prefix, df.irrelevant_word)
    ]

    assert not df.target_new_str.str.contains(" ").any()
    assert not df.target_true_str.str.contains(" ").any()
    assert not df.irrelevant_word.str.contains(" ").any()

    def tokenize(s: str):
        if isinstance(tokenizer, tiktoken.core.Encoding):
            return tokenizer.encode(s)
        else:
            return tokenizer(s, add_special_tokens=False).input_ids

    df = df[[len(tokenize(s)) == 1 for s in df.target_true_str]].reset_index(
        drop=True
    )

    return df
