import numpy as np
import torch as th
import transformer_lens as tl
from tuned_lens.plotting.prediction_trajectory import (
    PredictionTrajectory,
    ResidualComponent,
)

try:
    import transformer_lens as tl

    _transformer_lens_available = True
except ImportError:
    _transformer_lens_available = False

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Union
from numpy.typing import NDArray

from tuned_lens.nn.lenses import Lens
from tuned_lens.plotting import (
    TokenFormatter,
    TrajectoryLabels,
    TrajectoryStatistic,
)

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from tuned_lens.plotting import PredictionTrajectory


class PerHeadPredictionTrajectory(PredictionTrajectory):
    @classmethod
    def from_lens_and_cache(
        cls,
        lens: Lens,
        input_ids: Tensor,
        cache: ActivationCache,
        model_logits: Tensor,
        targets: Tensor | None = None,
        anti_targets: Tensor | None = None,
        residual_component: ResidualComponent = "resid_pre",
        attn_head: int | None = None,
        mask_input: bool = False,
    ) -> PredictionTrajectory:
        """Construct a prediction trajectory from a set of residual stream vectors.

        Args:
            lens: A lens to use to produce the predictions. Note this should be
                compatible with the model.
            model: A Hugging Face causal language model to use to produce
                the predictions.
            tokenizer: The tokenizer to use for decoding the input ids.
            input_ids: (seq_len) Ids that were input into the model.
            targets: (seq_len) the targets the model is should predict. Used
                for :meth:`cross_entropy` and :meth:`log_prob_diff` visualization.
            anti_targets: (seq_len) the incorrect label the model should not
                predict. Used for :meth:`log_prob_diff` visualization.
            residual_component: Name of the stream vector being visualized.
            mask_input: Whether to mask the input ids when computing the log probs.

        Returns:
            PredictionTrajectory constructed from the residual stream vectors.
        """
        with th.no_grad():
            input_ids_th = th.tensor(input_ids, dtype=th.int64, device=model.device)
            outputs = model(input_ids_th.unsqueeze(0), output_hidden_states=True)

        # Slice arrays the specified range
        model_log_probs = (
            outputs.logits[..., :].log_softmax(-1).squeeze().detach().cpu().numpy()
        )
        stream = list(outputs.hidden_states)

        input_ids_np = np.array(input_ids)
        targets_np = np.array(targets) if targets is not None else None
        anti_targets_np = np.array(anti_targets) if anti_targets is not None else None

        # Create the stream of log probabilities from the lens
        traj_log_probs = []
        for i, h in enumerate(stream[:-1]):
            logits = lens.forward(h, i)

            if mask_input:
                logits[..., input_ids_np] = -th.finfo(h.dtype).max

            traj_log_probs.append(
                logits.log_softmax(dim=-1).squeeze().detach().cpu().numpy()
            )

        # Add model predictions
        traj_log_probs.append(model_log_probs)

        return cls(
            tokenizer=tokenizer,
            log_probs=np.array(traj_log_probs),
            targets=targets_np,
            input_ids=input_ids_np,
            anti_targets=anti_targets_np,
        )
