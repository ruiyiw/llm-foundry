# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import torch
from torchmetrics import Metric
import json
import copy
from typing import List, Dict, Any


__all__ = [
    'PairwiseSearchAccuracy',
]


class PairwiseSearchAccuracy(Metric):
    """Computes exact match for structured JSON BasicSearchOutput with specific fields.
    
    This metric evaluates if the predicted JSON structure matches the ground truth JSON structure.
    In particular, it checks if both 'chosen_state' and 'chosen_action' fields match exactly.
    
    The model will be expected to produce a valid JSON object with the required fields.
    
    Adds metric state variables:
        correct (float): The number of instances where both fields in the prediction match the ground truth.
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
            'parsed_output': [],
            'parsed_label': [],
            'result': [],
            'error_type': [],
        }
    
    def _parse_json(self, text: str) -> Dict:
        """Try to parse a string as JSON and extract the required fields.
        
        Returns a dictionary with the parsed fields or an empty dict if parsing fails.
        """
        try:
            # Find JSON-like structure in the text
            # This handles cases where the model outputs additional text
            text = text.strip()
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = text[start_idx:end_idx + 1]
                json_obj = json.loads(json_str)
                
                # Check if required fields exist
                if 'chosen_state' in json_obj and 'chosen_action' in json_obj:
                    return {
                        'chosen_state': str(json_obj['chosen_state']).strip(),
                        'chosen_action': str(json_obj['chosen_action']).strip()
                    }
            
            return {}
        except json.JSONDecodeError:
            return {}
    
    def update(
        self,
        batch: Dict[str, Any],
        outputs: List[str],
        labels: List[str],
    ):
        """Update metric states based on predictions and ground truth.
        
        Args:
            batch: The input batch (not used in this metric)
            outputs: List of string outputs from the model, expected to contain JSON
            labels: List of ground truth strings, expected to contain JSON
        """
        metric_result_dict = copy.deepcopy(self.metric_result_dict)
        
        for output, label in zip(outputs, labels):
            # Parse predicted output and ground truth label
            parsed_output = self._parse_json(output)
            parsed_label = self._parse_json(label)
            
            metric_result_dict['parsed_output'].append(parsed_output)
            metric_result_dict['parsed_label'].append(parsed_label)
            
            # Check if parsing was successful
            if not parsed_output or not parsed_label:
                self.total += torch.tensor(1.0)
                metric_result_dict['result'].append(0)
                if not parsed_output:
                    metric_result_dict['error_type'].append('invalid_output_json')
                else:
                    metric_result_dict['error_type'].append('invalid_label_json')
                continue
            
            # Check if both fields match exactly
            if (parsed_output['chosen_state'] == parsed_label['chosen_state'] and
                parsed_output['chosen_action'] == parsed_label['chosen_action']):
                self.correct += torch.tensor(1.0)
                metric_result_dict['result'].append(1)
                metric_result_dict['error_type'].append(None)
            else:
                metric_result_dict['result'].append(0)
                if parsed_output['chosen_state'] != parsed_label['chosen_state']:
                    if parsed_output['chosen_action'] != parsed_label['chosen_action']:
                        metric_result_dict['error_type'].append('both_mismatch')
                    else:
                        metric_result_dict['error_type'].append('state_mismatch')
                else:
                    metric_result_dict['error_type'].append('action_mismatch')
            
            self.total += torch.tensor(1.0)
        
        return metric_result_dict
    
    def compute(self):
        """Compute the accuracy metric."""
        assert isinstance(self.correct, torch.Tensor)
        assert isinstance(self.total, torch.Tensor)
        return self.correct / self.total if self.total > 0 else torch.tensor(0.)