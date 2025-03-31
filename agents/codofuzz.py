from typing import Union
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from core.agent import BaseAgent

class CoDoFuzz(BaseAgent):
    def __init__(self, agent_seed, config, query_size=1, 
                 M=10, k=5, alpha=0.7, beta=0.3):
        super().__init__(agent_seed, config, query_size)
        self.M = M  # Number of probability bins
        self.k = k  # Max samples per cell
        self.alpha = alpha  # Coverage weight
        self.beta = beta  # Uncertainty weight
        
        # Initialize coverage matrix on CPU
        self.n_classes = config['dataset']['class_count']
        self.coverage_matrix = torch.zeros((self.n_classes, M), 
                                  dtype=torch.int32, device='cpu')

    def _get_cell(self, pred_class: Tensor, prob: Tensor):
        """Map prediction to coverage matrix cell"""
        bin_idx = torch.clamp((prob * self.M).long(), 0, self.M-1)
        return pred_class.cpu(), bin_idx.cpu()

    def predict(self, x_unlabeled: Tensor,
                x_labeled: Tensor, y_labeled: Tensor,
                per_class_instances: dict,
                budget: int, added_images: int,
                initial_test_acc: float, current_test_acc: float,
                classifier: Module, optimizer: Optimizer,
                sample_size=5000) -> list[int]:
        
        # Get predictions
        with torch.no_grad():
            classifier.eval()
            logits = self._predict(x_unlabeled, classifier)
            probs = torch.softmax(logits, dim=-1)
            max_probs, pred_classes = torch.max(probs, dim=-1)

        # Move to CPU for coverage calculations
        pred_classes_cpu = pred_classes.cpu()
        max_probs_cpu = max_probs.cpu()
        
        # Create a temporary coverage matrix
        temp_coverage = self.coverage_matrix.clone()
        

        # Calculate scores
        scores = []
        for idx in range(len(x_unlabeled)):
            # Uncertainty score (1 - confidence)
            prob = max_probs_cpu[idx]
            uncertainty_score = 1 - prob.item()
            total_score = self.beta * uncertainty_score
            scores.append(total_score)

        
        # Convert scores to tensor
        scores_tensor = torch.tensor(scores, device=x_unlabeled.device)
        sorted_indices = torch.argsort(scores_tensor, descending=True)

        selected = []
        taken = np.zeros(len(x_unlabeled), dtype=bool)
        
        for x in range(self.k):
            curr_score  = 1 / (x + 1e-6)

            if len(selected) >= self.query_size:
                break

            for idx in sorted_indices:
                if len(selected) >= self.query_size:
                    break
                if taken[idx]:
                    continue

                pc = pred_classes_cpu[idx]
                prob = max_probs_cpu[idx]
                row, col = self._get_cell(pc, prob)
                
                if temp_coverage[row, col] >= self.k:
                    continue

                # Get current cell count
                cell_count = temp_coverage[row, col].item()
                coverage_score = 1 / (cell_count + 1e-6)
                
                if coverage_score >= curr_score:
                    # Update the temporary coverage matrix
                    temp_coverage[row, col] += 1
                    selected.append(idx.item())
                    taken[idx] = True

        # Update the main coverage matrix and return selected
        self.coverage_matrix = temp_coverage

        if len(selected) < self.query_size:
            self.k = self.k + 2
        return selected