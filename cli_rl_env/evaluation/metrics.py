"""Metrics tracking for model evaluation."""

from typing import Dict, Any, List
from collections import defaultdict
import numpy as np


class EvaluationMetrics:
    """Track and compute evaluation metrics."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.results = []
        self.by_difficulty = defaultdict(list)
        self.by_type = defaultdict(list)
    
    def add_result(self, result: Dict[str, Any]):
        """Add a result to metrics.
        
        Args:
            result: Result dictionary from evaluation
        """
        self.results.append(result)
        
        # Track by difficulty
        difficulty = result.get('difficulty', 'unknown')
        self.by_difficulty[difficulty].append(result)
        
        # Track by scenario type
        scenario_type = result.get('scenario_type', 'unknown')
        self.by_type[scenario_type].append(result)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics.
        
        Returns:
            Dictionary of summary metrics
        """
        if not self.results:
            return {
                'total': 0,
                'passed': 0,
                'failed': 0,
                'pass_rate': 0.0
            }
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r['success'])
        failed = total - passed
        
        # Calculate averages
        avg_commands = np.mean([r['num_commands'] for r in self.results])
        avg_time = np.mean([r['time_seconds'] for r in self.results])
        
        # Pass rate by difficulty
        difficulty_stats = {}
        for diff, results in self.by_difficulty.items():
            diff_passed = sum(1 for r in results if r['success'])
            difficulty_stats[diff] = {
                'total': len(results),
                'passed': diff_passed,
                'pass_rate': (diff_passed / len(results)) * 100
            }
        
        # Pass rate by scenario type
        type_stats = {}
        for stype, results in self.by_type.items():
            type_passed = sum(1 for r in results if r['success'])
            type_stats[stype] = {
                'total': len(results),
                'passed': type_passed,
                'pass_rate': (type_passed / len(results)) * 100
            }
        
        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'pass_rate': (passed / total) * 100,
            'avg_commands': avg_commands,
            'avg_time': avg_time,
            'by_difficulty': difficulty_stats,
            'by_type': type_stats
        }
    
    def get_detailed_report(self) -> str:
        """Generate detailed text report.
        
        Returns:
            Formatted report string
        """
        summary = self.get_summary()
        
        report = []
        report.append("=" * 80)
        report.append("DETAILED EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall stats
        report.append(f"Total Examples: {summary['total']}")
        report.append(f"Passed: {summary['passed']} ({summary['pass_rate']:.1f}%)")
        report.append(f"Failed: {summary['failed']}")
        report.append(f"Average Commands: {summary['avg_commands']:.1f}")
        report.append(f"Average Time: {summary['avg_time']:.2f}s")
        report.append("")
        
        # By difficulty
        report.append("Performance by Difficulty:")
        report.append("-" * 80)
        for diff, stats in sorted(summary['by_difficulty'].items()):
            report.append(
                f"  {diff:15s}: {stats['passed']:3d}/{stats['total']:3d} "
                f"({stats['pass_rate']:5.1f}%)"
            )
        report.append("")
        
        # By scenario type
        report.append("Performance by Scenario Type:")
        report.append("-" * 80)
        for stype, stats in sorted(
            summary['by_type'].items(),
            key=lambda x: x[1]['pass_rate'],
            reverse=True
        ):
            report.append(
                f"  {stype:25s}: {stats['passed']:3d}/{stats['total']:3d} "
                f"({stats['pass_rate']:5.1f}%)"
            )
        report.append("")
        
        # Failed examples
        failed_results = [r for r in self.results if not r['success']]
        if failed_results:
            report.append(f"Failed Examples ({len(failed_results)}):")
            report.append("-" * 80)
            for r in failed_results[:10]:  # Show first 10
                report.append(
                    f"  {r['scenario_id']:20s} | "
                    f"{r['difficulty']:10s} | "
                    f"{r['scenario_type']:20s} | "
                    f"{r['num_commands']} cmds"
                )
            if len(failed_results) > 10:
                report.append(f"  ... and {len(failed_results) - 10} more")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def reset(self):
        """Reset all metrics."""
        self.results = []
        self.by_difficulty = defaultdict(list)
        self.by_type = defaultdict(list)

