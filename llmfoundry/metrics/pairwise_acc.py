# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import torch
# from torchmetrics import Metric
from llmfoundry.eval.metrics.nlp import InContextLearningMetric
import json
import copy
from typing import List


__all__ = [
    'PairwiseSearchAccuracy',
]


class PairwiseSearchAccuracy(InContextLearningMetric):
    """
    Customized accuracy metrics for https://github.com/pearls-lab/llm-search-textgame.

    Calculates the pairwise accuracy of output format {"chosen_state":..., "chosen_action":...}. The result is 1 iff. both items are the same as the ground truth.
    
    Adds metric state variables:
        correct (float): The number of instances where both state and action matched the target.
        total (float): The number of total instances that were predicted.
    
    Args:
        dist_sync_on_step (bool, optional): Synchronize metric state across processes at
            each forward() before returning the value at the step. Default: ``False``.
    """

    # Make torchmetrics call update only once
    full_state_update = False
    
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state(
            'correct',
            default=torch.tensor(0.),
            dist_reduce_fx='sum',
        )
        self.add_state('total', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.metric_result_dict = {
            'input': [],
            'label_state': [],
            'label_action': [],
            'output_state': [],
            'output_action': [],
            'correct': [],
        }

    def update(self, batch: dict, outputs: torch.Tensor, labels: torch.Tensor):
        """Update the metric state with a new batch of data.
        
        Args:
            batch (dict): The input batch data.
            outputs (torch.Tensor): The model outputs.
            labels (torch.Tensor): The target outputs.
            
        Returns:
            dict: The updated metric result dictionary.
        """
        metric_result_dict = copy.deepcopy(self.metric_result_dict)
        
        # Convert tensor outputs to string predictions
        # Assuming outputs are logits that need to be decoded to text
        output_texts = self._decode_outputs(batch, outputs)
        label_texts = self._decode_labels(batch, labels)
        
        for batch_idx in range(len(output_texts)):
            try:
                output_json = json.loads(output_texts[batch_idx])
                label_json = json.loads(label_texts[batch_idx])
                
                output_state = output_json.get('chosen_state', '')
                output_action = output_json.get('chosen_action', '')
                label_state = label_json.get('chosen_state', '')
                label_action = label_json.get('chosen_action', '')
                
                # Record inputs and outputs
                metric_result_dict['input'].append(batch.get('input_ids', [])[batch_idx])
                metric_result_dict['label_state'].append(label_state)
                metric_result_dict['label_action'].append(label_action)
                metric_result_dict['output_state'].append(output_state)
                metric_result_dict['output_action'].append(output_action)
                
                # Check correctness (both state and action must match)
                both_correct = (output_state == label_state) and (output_action == label_action)
                
                # Update metrics
                if both_correct:
                    self.correct += torch.tensor(1.0)
                
                # Record correctness
                metric_result_dict['correct'].append(int(both_correct))
                
                # Increment total
                self.total += torch.tensor(1.0)
                
            except json.JSONDecodeError:
                # Handle invalid JSON in outputs or labels
                metric_result_dict['input'].append(batch.get('input_ids', [])[batch_idx])
                metric_result_dict['label_state'].append('ERROR')
                metric_result_dict['label_action'].append('ERROR')
                metric_result_dict['output_state'].append('ERROR')
                metric_result_dict['output_action'].append('ERROR')
                metric_result_dict['correct'].append(0)
                self.total += torch.tensor(1.0)
        
        return metric_result_dict
        

    def _decode_outputs(self, batch: dict, outputs: torch.Tensor) -> List[str]:
        """Decode model outputs to strings containing JSON.
        
        This method needs to be implemented based on your tokenizer and model output format.
        The implementation below is a placeholder.
        
        Args:
            batch (dict): The input batch.
            outputs (torch.Tensor): The model outputs.
            
        Returns:
            List[str]: Decoded outputs as strings.
        """
        # Placeholder implementation
        # In practice, you would use your tokenizer to decode the outputs
        # Example with a tokenizer (you would replace this with your actual tokenizer):
        # return [tokenizer.decode(output) for output in outputs]
        
        # For now, assuming outputs are already decoded to strings
        if isinstance(outputs, list) and isinstance(outputs[0], str):
            return outputs
        
        # If outputs are tensors, you need to decode them
        # This is just a placeholder - replace with actual decoding logic
        return [""] * len(batch['input_ids'])
    
    def _decode_labels(self, batch: dict, labels: torch.Tensor) -> List[str]:
        """Decode label tensors to strings containing JSON.
        
        This method needs to be implemented based on your tokenizer and label format.
        The implementation below is a placeholder.
        
        Args:
            batch (dict): The input batch.
            labels (torch.Tensor): The target labels.
            
        Returns:
            List[str]: Decoded labels as strings.
        """
        # Placeholder implementation
        # In practice, you would use your tokenizer to decode the labels
        # Example with a tokenizer (you would replace this with your actual tokenizer):
        # return [tokenizer.decode(label) for label in labels]
        
        # For now, assuming labels are already decoded to strings
        if isinstance(labels, list) and isinstance(labels[0], str):
            return labels
        
        # If labels are tensors, you need to decode them
        # This is just a placeholder - replace with actual decoding logic
        return [""] * len(batch['input_ids'])
    
    def compute(self):
        """Compute the accuracy metric based on accumulated state.
        
        Returns:
            float: The proportion of instances where both state and action matched.
        """
        assert isinstance(self.correct, torch.Tensor)
        assert isinstance(self.total, torch.Tensor)
        
        # Avoid division by zero
        if self.total == 0:
            return torch.tensor(0.)
        
        return self.correct / self.total