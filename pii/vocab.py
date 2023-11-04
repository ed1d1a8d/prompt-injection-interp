import collections

from transformers import AutoTokenizer
from jaxtyping import Float
import torch



class VocabEquivalenceMap:
    def __init__(self, tokenizer: AutoTokenizer):
        self.lcs_str_to_toks = collections.defaultdict(list)
        for tok in range(tokenizer.vocab_size):
            lcs_str = tokenizer.batch_decode([tok], clean_up_tokenization_spaces=False)[0].lower()
            self.lcs_str_to_toks[lcs_str].append(tok)

        self.tok_to_equivalent_toks: dict[int, list[int]] = {}
        for toks in self.lcs_str_to_toks.values():
            for tok in toks:
                self.tok_to_equivalent_toks[tok] = toks

    def get_equivalent_tokens(self, x: int | str) -> list[int]:
        if isinstance(x, str):
            return self.lcs_str_to_toks[x.lower()]
        else:
            return self.tok_to_equivalent_toks[x]

    def p_correct(
        self,
        probs: Float[torch.Tensor, "... vocab"],
        correct_answer: str,
    ) -> Float[torch.Tensor, "..."]:
        tot = 0
        for tok in self.get_equivalent_tokens(correct_answer):
            tot += probs[..., tok]
        return tot
