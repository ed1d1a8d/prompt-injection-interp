import collections

from transformer_lens import HookedTransformer


class VocabEquivalenceMap:
    def __init__(self, tl_model: HookedTransformer):
        self.lcs_str_to_toks = collections.defaultdict(list)
        for tok in range(tl_model.cfg.d_vocab):
            lcs_str = tl_model.to_single_str_token(tok).lower()
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
