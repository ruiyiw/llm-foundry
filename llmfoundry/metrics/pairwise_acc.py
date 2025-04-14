# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import torch
from torchmetrics import Metric
from transformers import PreTrainedTokenizerBase

import json
from typing import Dict, Any, Optional


__all__ = [
    'PairwiseSearchAccuracy',
    'PairwiseTrajectoryAccuracy'
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
        # Convert logits to predicted token indices if necessary
        if preds.dim() > 2:
            preds = torch.argmax(preds, dim=-1)
        
        batch_size = preds.size(0)
        
        for i in range(batch_size):
            # Decode the target and prediction tensors to strings
            # Mask out the ignored indices for the target
            target_mask = (target[i] != self.ignore_index)
            target_tokens = target[i][target_mask]

            preds_mask = (preds[i] != self.ignore_index)
            preds_tokens = preds[i][preds_mask]
            
            target_str = self.tokenizer.decode(target_tokens, skip_special_tokens=True)
            preds_str = self.tokenizer.decode(preds_tokens, skip_special_tokens=True)
            
            # Parse the strings as JSON
            target_json = self._parse_json(target_str)
            preds_json = self._parse_json(preds_str)
            
            # If prediction cannot be jsonized, assign zero score
            if preds_json is None:
                self.total += 1
                continue
                
            # Check if both required fields exist
            if (target_json and 'chosen_state' in target_json and 'chosen_action' in target_json and
                preds_json and 'chosen_state' in preds_json and 'chosen_action' in preds_json):
                
                # Compare the values of chosen_state and chosen_action
                if (target_json['chosen_state'] == preds_json['chosen_state'] and 
                    target_json['chosen_action'] == preds_json['chosen_action']):
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
    


class PairwiseTrajectoryAccuracy(Metric):
    """
    Computes match accuracy for structured JSON trajectories.
    
    This metric evaluates if the predicted JSON trajectory matches the ground truth JSON trajectory.
    A trajectory is a list of dictionaries with 'type' and 'content' fields.
    The metric compares the 'content' field of each element in the trajectory in order.
    
    The model will be expected to produce a valid JSON object with the required fields.
    
    Adds metric state variables:
        correct_steps (float): The number of trajectory steps that match between prediction and ground truth.
        total_steps (float): The total number of steps in the ground truth trajectories.
    
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
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.add_state(
            'correct_steps',
            default=torch.tensor(0),
            dist_reduce_fx='sum',
        )
        self.add_state(
            'total_steps',
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
        # Convert logits to predicted token indices if necessary
        if preds.dim() > 2:
            preds = torch.argmax(preds, dim=-1)
        
        batch_size = preds.size(0)
        
        for i in range(batch_size):
            # Decode the target and prediction tensors to strings
            # Mask out the ignored indices for the target
            target_mask = (target[i] != self.ignore_index)
            target_tokens = target[i][target_mask]

            preds_mask = (preds[i] != self.ignore_index)
            preds_tokens = preds[i][preds_mask]
            
            target_str = self.tokenizer.decode(target_tokens, skip_special_tokens=True)
            preds_str = self.tokenizer.decode(preds_tokens, skip_special_tokens=True)
            
            # Parse the strings as JSON
            target_json = self._parse_json(target_str)
            preds_json = self._parse_json(preds_str)
            
            # If either cannot be jsonized, skip this example
            if target_json is None or preds_json is None:
                continue
            
            print(target_json)
            print(preds_json)
            
            # Check if both have the 'trajectory' field with a list
            if ('trajectory' in target_json and isinstance(target_json['trajectory'], list) and
                'trajectory' in preds_json and isinstance(preds_json['trajectory'], list)):
                
                target_traj = target_json['trajectory']
                pred_traj = preds_json['trajectory']
                
                # Get the number of steps to compare (limited by target length)
                target_length = len(target_traj)
                compare_length = min(len(target_traj), len(pred_traj))
                
                # Count correct steps
                correct = 0
                for j in range(compare_length):
                    # Check if both elements have 'content' field
                    if ('content' in target_traj[j] and 'content' in pred_traj[j] and
                        target_traj[j]['content'] == pred_traj[j]['content']):
                        correct += 1
                
                # Update counters
                self.correct_steps += correct
                self.total_steps += target_length
            else:
                # If the expected structure is not found, count as all incorrect
                # We still need to count the target steps for proper denominator
                if 'trajectory' in target_json and isinstance(target_json['trajectory'], list):
                    self.total_steps += len(target_json['trajectory'])

    def compute(self) -> torch.Tensor:
        """Aggregate the state over all processes to compute the metric.

        Returns:
            The accuracy as a :class:`~torch.Tensor`.
        """
        assert isinstance(self.correct_steps, torch.Tensor)
        assert isinstance(self.total_steps, torch.Tensor)
        
        return self.correct_steps.float() / self.total_steps if self.total_steps > 0 else torch.tensor(0.0)