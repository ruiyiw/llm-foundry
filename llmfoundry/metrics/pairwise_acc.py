# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import torch
from torchmetrics import Metric
from transformers import PreTrainedTokenizerBase

import json
from typing import Dict, Any, Optional


__all__ = [
    'PairwiseSearchAccuracy',
]


class PairwiseSearchAccuracy(Metric):
    """
    Computes exact match for structured JSON BasicSearchOutput with specific fields.
    
    This metric evaluates if the predicted JSON structure matches the ground truth JSON structure.
    In particular, it checks if both 'chosen_state' and 'chosen_action' fields match exactly.
    
    The model will be expected to produce a valid JSON object with the required fields.
    
    Adds metric state variables:
        correct (float): The number of instances where both fields in the prediction match the ground truth.
        total (float): The number of total instances that were predicted.
    
    Args:
        tokenizer (PreTrainedTokenizerBase): The tokenizer used to decode the token IDs into text.
        ignore_index (int, optional): The value in the target to be ignored during evaluation.
            Default: ``-100``.
        dist_sync_on_step (bool, optional): Synchronize metric state across processes at
            each forward() before returning the value at the step. Default: ``False``.
    """

    # Ensures torchmetrics calls update only once
    full_state_update = False

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        ignore_index: int = -100,
        dist_sync_on_step: bool = False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.add_state(
            'correct',
            default=torch.tensor(0),
            dist_reduce_fx='sum',
        )
        self.add_state(
            'total',
            default=torch.tensor(0),
            dist_reduce_fx='sum',
        )

    def _parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse a string as JSON and return the resulting dictionary.
        
        Returns None if the string cannot be parsed as valid JSON.
        """
        try:
            data = json.loads(text)
            return data
        except json.JSONDecodeError:
            return None

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Updates the internal state with results from a new batch.

        Args:
            preds (~torch.Tensor): The predictions from the model, a Tensor of token IDs.
            target (~torch.Tensor): A Tensor of ground-truth token IDs.
        """
        print("*** entrypoint: pairwise_accuracy ***")
        print(preds.shape)
        # Convert logits to predicted token indices if necessary
        if preds.dim() > 2:
            preds = torch.argmax(preds, dim=-1)
        
        batch_size = preds.size(0)
        print(batch_size)
        
        for i in range(batch_size):
            # Decode the target and prediction tensors to strings
            # Mask out the ignored indices for the target
            target_mask = target[i] != self.ignore_index
            target_tokens = target[i][target_mask]

            preds_mask = preds[i] != self.ignore_index
            preds_tokens = preds[i][preds_mask]
            
            target_str = self.tokenizer.decode(target_tokens, skip_special_tokens=True)
            pred_str = self.tokenizer.decode(preds_tokens, skip_special_tokens=True)

            print(target_str)
            print(pred_str)
            
            # Parse the strings as JSON
            target_json = self._parse_json(target_str)
            pred_json = self._parse_json(pred_str)
            
            # If prediction cannot be jsonized, assign zero score
            if pred_json is None:
                self.total += 1
                continue
                
            # Check if both required fields exist
            if (target_json and 'chosen_state' in target_json and 'chosen_action' in target_json and
                pred_json and 'chosen_state' in pred_json and 'chosen_action' in pred_json):
                
                # Compare the values of chosen_state and chosen_action
                if (target_json['chosen_state'] == pred_json['chosen_state'] and 
                    target_json['chosen_action'] == pred_json['chosen_action']):
                    self.correct += 1
            
            self.total += 1

    def compute(self) -> torch.Tensor:
        """Aggregate the state over all processes to compute the metric.

        Returns:
            The accuracy as a :class:`~torch.Tensor`.
        """
        assert isinstance(self.correct, torch.Tensor)
        assert isinstance(self.total, torch.Tensor)
        
        return self.correct.float() / self.total if self.total > 0 else torch.tensor(0.0)