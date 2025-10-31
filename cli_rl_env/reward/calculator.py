"""Calculate rewards for RL training."""

from typing import Dict, Any, List
import numpy as np


class RewardCalculator:
    """Calculate normalized rewards (0-1 range) based on verification and time performance."""
    
    def __init__(
        self,
        time_penalty_weight: float = 0.1,
        regression_penalty_weight: float = 0.3
    ):
        """Initialize reward calculator.
        
        Args:
            time_penalty_weight: Weight for time penalty (0-1)
            regression_penalty_weight: Weight for regression penalty (0-1)
        """
        self.time_penalty_weight = time_penalty_weight
        self.regression_penalty_weight = regression_penalty_weight
    
    def calculate_reward(
        self,
        verification_results: Dict[str, Any],
        actual_time: float,
        estimated_time: float,
        initial_test_results: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Calculate total reward in [0, 1] range.
        
        Args:
            verification_results: Results from verification
            actual_time: Actual execution time
            estimated_time: Time estimated by model
            initial_test_results: Initial test results before changes (to detect regressions)
            
        Returns:
            Dict with reward breakdown, all values in [0, 1] range
        """
        # Calculate base reward from verification (0-1)
        base_reward = self._calculate_base_reward(verification_results)
        
        # Calculate time penalty (0-1, where 1 = no penalty)
        time_score = self._calculate_time_score(actual_time, estimated_time)
        
        # Calculate regression penalty (0-1, where 1 = no penalty)
        regression_score = 1.0
        if initial_test_results:
            regression_score = self._calculate_regression_score(
                initial_test_results,
                verification_results
            )
        
        # Combine scores with weights
        # Base reward is primary (weighted heavily)
        # Time and regression are penalties that reduce the reward
        total_reward = base_reward * (
            1.0 - self.time_penalty_weight * (1.0 - time_score)
        ) * (
            1.0 - self.regression_penalty_weight * (1.0 - regression_score)
        )
        
        # Ensure final reward is in [0, 1]
        total_reward = np.clip(total_reward, 0.0, 1.0)
        
        return {
            'total_reward': float(total_reward),
            'base_reward': float(base_reward),
            'time_score': float(time_score),
            'regression_score': float(regression_score),
            'actual_time': actual_time,
            'estimated_time': estimated_time,
            'breakdown': {
                'verification_score': float(base_reward),
                'time_penalty': float(1.0 - time_score),
                'regression_penalty': float(1.0 - regression_score)
            }
        }
    
    def _calculate_base_reward(self, verification_results: Dict[str, Any]) -> float:
        """Calculate base reward from verification results.
        
        Args:
            verification_results: Dict with test/lint/text match results
            
        Returns:
            Base reward in [0, 1] range
        """
        scores = []
        weights = []
        
        # Test results (weight: 0.7 - most important)
        if 'test_results' in verification_results:
            test_res = verification_results['test_results']
            if test_res.get('total', 0) > 0:
                test_score = test_res['passed'] / test_res['total']
                scores.append(test_score)
                weights.append(0.7)
            elif test_res.get('success', False):
                scores.append(1.0)
                weights.append(0.7)
            else:
                # Tests were present but failed or could not collect
                scores.append(0.0)
                weights.append(0.7)
        
        # Linting results (weight: 0.2)
        if 'lint_results' in verification_results:
            lint_res = verification_results['lint_results']
            if not lint_res.get('skipped', False):
                error_count = lint_res.get('error_count', 0)
                if error_count == 0:
                    lint_score = 1.0
                else:
                    # Deduct based on error count, but cap at minimum 0.5
                    lint_score = max(0.5, 1.0 - (error_count * 0.05))
                scores.append(lint_score)
                weights.append(0.2)
        
        # Text matching results (weight: 0.1)
        if 'text_match_results' in verification_results:
            text_matches = verification_results['text_match_results']
            if isinstance(text_matches, list) and len(text_matches) > 0:
                match_score = sum(1 for m in text_matches if m.get('success', False)) / len(text_matches)
                scores.append(match_score)
                weights.append(0.1)
            elif isinstance(text_matches, dict) and text_matches.get('success', False):
                scores.append(1.0)
                weights.append(0.1)

        # Permissions verification (weight: 0.1 when expectations exist)
        if 'permissions_verification' in verification_results:
            perm = verification_results['permissions_verification']
            if perm.get('has_expectations', False):
                perm_score = 1.0 if perm.get('success', False) else 0.0
                scores.append(perm_score)
                weights.append(0.1)
        
        # Execution verification (weight: 0.05 - baseline, lowest priority)
        # Only used if no other verification exists
        if 'execution_verification' in verification_results:
            exec_verify = verification_results['execution_verification']
            # If we have no other verification, execution verification is important
            if not scores:
                # No other verification - execution verification is critical
                exec_score = 1.0 if exec_verify.get('success', False) else 0.0
                scores.append(exec_score)
                weights.append(1.0)  # Full weight since it's the only verification
            else:
                # We have other verification - execution is just a bonus
                exec_score = 1.0 if exec_verify.get('success', False) else 0.5
                scores.append(exec_score)
                weights.append(0.05)  # Small weight as supplementary verification
        
        # Calculate weighted average
        if scores:
            # Normalize weights to sum to 1
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            
            weighted_score = sum(s * w for s, w in zip(scores, normalized_weights))
            return np.clip(weighted_score, 0.0, 1.0)
        else:
            # No verification ran at all - return 0
            return 0.0
    
    def _calculate_time_score(self, actual_time: float, estimated_time: float) -> float:
        """Calculate time score (1.0 = perfect, 0.0 = terrible).
        
        Args:
            actual_time: Actual execution time
            estimated_time: Estimated time from model
            
        Returns:
            Time score in [0, 1] range (1.0 = no penalty)
        """
        if estimated_time <= 0:
            # Invalid estimate, give neutral score
            return 0.5
        
        # Calculate ratio of actual to estimated
        time_ratio = actual_time / estimated_time
        
        if time_ratio <= 1.0:
            # Finished within estimate or early - perfect score
            return 1.0
        else:
            # Exceeded estimate - penalty based on how much over
            # If 2x over estimate: score = 0.5
            # If 3x over estimate: score = 0.33
            # etc.
            penalty_ratio = 1.0 / time_ratio
            return np.clip(penalty_ratio, 0.0, 1.0)
    
    def _calculate_regression_score(
        self,
        initial_results: Dict[str, Any],
        final_results: Dict[str, Any]
    ) -> float:
        """Calculate regression score (1.0 = no regression, 0.0 = broke everything).
        
        Args:
            initial_results: Test results before changes
            final_results: Test results after changes
            
        Returns:
            Regression score in [0, 1] range (1.0 = no penalty)
        """
        # Check if we broke tests that were passing
        initial_passed = initial_results.get('test_results', {}).get('passed', 0)
        final_passed = final_results.get('test_results', {}).get('passed', 0)
        
        if initial_passed == 0:
            # No tests were passing initially, can't regress
            return 1.0
        
        if final_passed < initial_passed:
            # Broke some tests that were passing
            tests_broken = initial_passed - final_passed
            regression_ratio = tests_broken / initial_passed
            
            # Return score (1.0 = no regression, 0.0 = broke all)
            return 1.0 - regression_ratio
        else:
            # No regression (or improvement!)
            return 1.0
    
    def calculate_partial_credit(
        self,
        expected_commands: int,
        actual_commands: int,
        verification_results: Dict[str, Any]
    ) -> float:
        """Calculate efficiency bonus/penalty.
        
        Args:
            expected_commands: Expected number of commands
            actual_commands: Actual number of commands used
            verification_results: Verification results
            
        Returns:
            Efficiency score in [0, 1] range
        """
        test_res = verification_results.get('test_results', {})
        
        # Only reward efficiency if task is complete
        if test_res.get('success', False):
            if actual_commands <= expected_commands:
                # Efficient - bonus
                efficiency = min(expected_commands / max(actual_commands, 1), 1.5)
                return min(efficiency, 1.0)
            else:
                # Inefficient but correct - small penalty
                efficiency = expected_commands / actual_commands
                return max(efficiency, 0.7)  # Cap penalty at 30%
        
        # Task not complete - return neutral
        return 1.0
