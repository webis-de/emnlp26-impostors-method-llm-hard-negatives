# Copyright 2024 Janek Bevendorff, Webis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import batched
import typing as t
from sys import orig_argv

import numpy as np
import torch
from tqdm import tqdm

from genai_detection.detectors.detector_base import DetectorBase
from genai_detection.util import *

__all__ = ["LLMUnmasking"]


class LLMUnmasking(DetectorBase):
    """
    Deep Unmasking LLM detector.
    """

    def __init__(
        self,
        base_model="mistralai/Mistral-7B-v0.1",
        device: TorchDeviceMapType = "auto",
        verbose=True,
        n_remove=64,
        batch_size=1,
        **base_model_args
    ):
        """
        :param base_model: base language model
        :param device: base model device
        :param verbose: show progress bar
        :param base_model_args: additional base model arguments
        """

        self.verbose = verbose
        self.n_remove = n_remove
        self.batch_size = batch_size
        self.base_model, self.base_tokenizer = load_model(
            base_model, task_type="CAUSAL_LM", device_map=device, **base_model_args
        )

    # Elimination
    # @torch.inference_mode()
    # def get_curves(self, text: t.Iterable[str]) -> np.ndarray:
    #     curves = []
    #     for t in tqdm(text, desc='Generating curves', unit='curve',
    #                   disable=not self.verbose, leave=False):
    #         encoding = tokenize_sequences(t, self.base_tokenizer, self.base_model.device, 512)
    #         c = np.zeros(self.n_rounds)
    #         for i in tqdm(range(self.n_rounds), desc='Estimating log likelihoods',
    #                       disable=not self.verbose, leave=False):
    #             ll = batch_seq_log_likelihood(self.base_model, encoding, 1,
    #                                           verbose=False, aggregate=False).squeeze().cpu()
    #             ll_argsort = ll.argsort()[:self.n_remove]
    #             c[i] = len(ll) / ll.sum()
    #             mask = torch.ones(encoding.input_ids.squeeze().shape, dtype=torch.bool)
    #             mask[ll_argsort] = False
    #             mask[ll_argsort + 1] = False
    #             encoding.data = {k: v[:, mask] for k, v in encoding.data.items()}
    #         curves.append(c)
    #     return np.array(curves)

    # Argmin token replacement
    # @torch.inference_mode()
    # def get_curves(self, text: t.Iterable[str], n_rounds) -> np.ndarray:
    #     curves = []
    #     import time
    #     start = time.monotonic()
    #     for batch in tqdm(batched(text, self.batch_size), desc='Generating curve points', unit=' batches',
    #                       disable=not self.verbose, leave=False):
    #         encoding = tokenize_sequences(batch, self.base_tokenizer, self.base_model.device, 256)
    #         c = np.zeros((len(batch), n_rounds + 1))
    #         for i in tqdm(range(n_rounds + 1), desc='Estimating log likelihoods',
    #                       disable=not self.verbose, leave=False):
    #             logits = self.base_model(**encoding, labels=encoding.input_ids).logits.detach()
    #             ll = seq_label_log_rank(logits, encoding.input_ids, encoding.attention_mask,
    #                                          aggregate=False).squeeze(-1)
    #             ll_argsort = ll.argsort(dim=-1)[:, :self.n_remove]
    #             ll = (encoding.attention_mask[:, 1:].sum(dim=-1) / ll.sum(dim=-1)).float().cpu()
    #             c[:, i] = ll.nan_to_num(0)
    #             mask = torch.zeros(encoding.input_ids.shape, dtype=torch.bool, device=logits.device)
    #             mask_lshift = mask.clone()
    #             ar = torch.arange(mask.shape[0], device=logits.device).unsqueeze(-1)
    #             mask[ar, ll_argsort] = True
    #             mask_lshift[ar, ll_argsort - 1] = True
    #             encoding.input_ids[mask_lshift] = logits[mask].argmin(dim=-1)
    #         curves.extend(c)
    #     print(f'{time.monotonic() - start:.2f} seconds')
    #     return np.array(curves)

    # <UNK> replacement
    # @torch.inference_mode()
    # def get_curves(self, text: t.Iterable[str], n_rounds) -> np.ndarray:
    #     curves = []
    #     for batch in tqdm(batched(text, self.batch_size), desc='Generating curve points', unit=' batches',
    #                       disable=not self.verbose, leave=False):
    #         encoding = tokenize_sequences(batch, self.base_tokenizer, self.base_model.device, 512)
    #         c = np.zeros((len(batch), n_rounds + 1))
    #         for i in tqdm(range(n_rounds + 1), desc='Estimating log likelihoods',
    #                       disable=not self.verbose, leave=False):
    #             ll = batch_seq_log_likelihood(self.base_model, encoding, encoding.input_ids.shape[0],
    #                                           verbose=False, aggregate=False).squeeze(-1)
    #             ll_argsort = ll.argsort(dim=-1)[:, :self.n_remove]
    #             ll = (encoding.attention_mask[:, 1:].sum(dim=-1) / ll.sum(dim=-1)).float().cpu()
    #             c[:, i] = ll.nan_to_num(0)
    #             mask = torch.zeros(encoding.input_ids.shape, dtype=torch.bool)
    #             mask[torch.arange(mask.shape[0]).unsqueeze(-1), ll_argsort] = True
    #             encoding.input_ids[mask] = self.base_tokenizer.unk_token_id
    #             # encoding.data = {k: v[~mask].reshape(mask.shape[0], -1) for k, v in encoding.data.items()}
    #         curves.extend(c)
    #     return np.array(curves)

    # Concept drift
    @torch.inference_mode()
    def get_curves(self, text: t.Iterable[str], n_rounds) -> np.ndarray:
        curves = []
        for batch in tqdm(
            batched(text, self.batch_size),
            desc="Generating curve points",
            unit=" batches",
            disable=not self.verbose,
            leave=False,
        ):
            encoding = tokenize_sequences(
                batch, self.base_tokenizer, self.base_model.device, 256
            )
            orig_ids = encoding.input_ids.clone()
            c = np.zeros((len(batch), n_rounds + 1))
            for i in tqdm(
                range(n_rounds + 1),
                desc="Predicting next tokens",
                disable=not self.verbose,
                leave=False,
            ):
                logits = self.base_model(**encoding).logits.detach().clone()
                encoding.data["input_ids"] = logits.argmax(dim=-1)
                c[:, i] = (
                    seq_label_cross_entropy(logits, orig_ids, encoding.attention_mask)
                    .float()
                    .cpu()
                )
            curves.extend(c)
        return np.array(curves)
