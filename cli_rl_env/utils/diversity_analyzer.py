"""Analyze and ensure diversity in training datasets."""

import json
import re
from typing import Dict, List, Set, Any
from collections import defaultdict, Counter


class DiversityAnalyzer:
    """Analyze command diversity in training datasets."""
    
    # All supported commands grouped by category
    COMMAND_CATEGORIES = {
        'file_viewing': {'cat', 'head', 'tail', 'less', 'more'},
        'file_search': {'grep', 'find'},
        'text_processing': {'sed', 'awk', 'cut', 'tr', 'sort', 'uniq', 'wc'},
        'file_operations': {'cp', 'mv', 'rm', 'mkdir', 'touch'},
        'text_output': {'echo', 'printf', 'tee'},
        'comparison': {'diff', 'cmp', 'comm', 'patch'},
        'navigation': {'cd', 'pwd', 'ls'},
        'git': {'git'},
    }
    
    # Flatten for easy lookup
    ALL_COMMANDS = set()
    for commands in COMMAND_CATEGORIES.values():
        ALL_COMMANDS.update(commands)
    
    def __init__(self, underrepresented_threshold: float = 0.05):
        """Initialize analyzer.
        
        Args:
            underrepresented_threshold: Threshold below which commands are considered
                underrepresented (default 0.05 = 5% of scenarios)
                
        Raises:
            ValueError: If threshold not in [0.0, 1.0]
        """
        if not 0.0 <= underrepresented_threshold <= 1.0:
            raise ValueError(
                f"underrepresented_threshold must be between 0.0 and 1.0, "
                f"got {underrepresented_threshold}"
            )
        
        self.command_counts = Counter()
        self.category_counts = Counter()
        self.scenario_types = Counter()
        self.total_scenarios = 0
        self.underrepresented_threshold = underrepresented_threshold
        
    def analyze_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Analyze a dataset file and return diversity metrics.
        
        Args:
            dataset_path: Path to JSON dataset file
            
        Returns:
            Dictionary with diversity statistics
        """
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        self.total_scenarios = len(data)
        
        for example in data:
            # Track scenario types
            if 'metadata' in example and 'scenario_type' in example['metadata']:
                self.scenario_types[example['metadata']['scenario_type']] += 1
            
            # Track commands in task description
            if 'task_description' in example:
                self._analyze_task_commands(example['task_description'])
        
        return self._generate_report()
    
    def _analyze_commands(self, commands: List[str]):
        """Analyze a list of commands."""
        for cmd in commands:
            # Extract base command
            parts = cmd.strip().split()
            if parts:
                base_cmd = parts[0]
                
                # Handle pipes
                if '|' in cmd:
                    pipe_parts = cmd.split('|')
                    for part in pipe_parts:
                        sub_parts = part.strip().split()
                        if sub_parts:
                            self.command_counts[sub_parts[0]] += 1
                            self._update_category(sub_parts[0])
                else:
                    self.command_counts[base_cmd] += 1
                    self._update_category(base_cmd)
    
    def _analyze_task_commands(self, task_description: str):
        """Extract commands mentioned in task descriptions using regex.
        
        This method uses regex patterns to robustly extract command names from:
        - Plain text mentions
        - Code blocks
        - Command examples with arguments
        - Inline code formatting
        """
        # Create pattern that matches any known command as a whole word
        # Sort by length (longest first) to match 'grep' before 'rep' in 'grep'
        sorted_commands = sorted(self.ALL_COMMANDS, key=len, reverse=True)
        pattern = r'\b(' + '|'.join(re.escape(cmd) for cmd in sorted_commands) + r')\b'
        
        # Find all command matches (case-insensitive)
        matches = re.findall(pattern, task_description.lower())
        
        # Count each match
        for cmd in matches:
            if cmd in self.ALL_COMMANDS:
                self.command_counts[cmd] += 1
                self._update_category(cmd)
    
    def _update_category(self, command: str):
        """Update category counts for a command."""
        for category, commands in self.COMMAND_CATEGORIES.items():
            if command in commands:
                self.category_counts[category] += 1
                break
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate diversity report."""
        # Calculate coverage percentages
        used_commands = set(self.command_counts.keys())
        total_commands = len(self.ALL_COMMANDS)
        coverage = len(used_commands) / total_commands * 100
        
        # Find missing commands
        missing_commands = self.ALL_COMMANDS - used_commands
        
        # Calculate category coverage
        category_coverage = {}
        for category, commands in self.COMMAND_CATEGORIES.items():
            used = used_commands & commands
            total = len(commands)
            category_coverage[category] = {
                'used': len(used),
                'total': total,
                'percentage': len(used) / total * 100 if total > 0 else 0,
                'missing': list(commands - used)
            }
        
        # Find underrepresented commands (below threshold)
        threshold = self.total_scenarios * self.underrepresented_threshold
        underrepresented = {
            cmd: count for cmd, count in self.command_counts.items()
            if count < threshold
        }
        
        return {
            'total_scenarios': self.total_scenarios,
            'command_coverage': {
                'percentage': coverage,
                'used_commands': len(used_commands),
                'total_commands': total_commands,
                'missing_commands': sorted(list(missing_commands))
            },
            'command_counts': dict(self.command_counts.most_common()),
            'category_coverage': category_coverage,
            'category_counts': dict(self.category_counts),
            'scenario_types': dict(self.scenario_types),
            'underrepresented_commands': underrepresented,
            'recommendations': self._generate_recommendations(
                missing_commands, underrepresented, category_coverage
            )
        }
    
    def _generate_recommendations(
        self,
        missing: Set[str],
        underrepresented: Dict[str, int],
        category_coverage: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations to improve diversity."""
        recommendations = []
        
        if missing:
            recommendations.append(
                f"Add scenarios using: {', '.join(sorted(missing))}"
            )
        
        if underrepresented:
            top_under = sorted(underrepresented.items(), key=lambda x: x[1])[:5]
            commands = [cmd for cmd, _ in top_under]
            recommendations.append(
                f"Increase usage of: {', '.join(commands)}"
            )
        
        # Check categories with low coverage
        for category, stats in category_coverage.items():
            if stats['percentage'] < 50 and stats['missing']:
                recommendations.append(
                    f"Category '{category}' only {stats['percentage']:.1f}% covered. "
                    f"Add: {', '.join(stats['missing'][:3])}"
                )
        
        return recommendations
    
    def print_report(self, report: Dict[str, Any]):
        """Print a formatted diversity report."""
        print("\n" + "="*80)
        print("DATASET DIVERSITY REPORT")
        print("="*80)
        
        print(f"\nTotal Scenarios: {report['total_scenarios']}")
        
        print(f"\n📊 Command Coverage: {report['command_coverage']['percentage']:.1f}%")
        print(f"   Used: {report['command_coverage']['used_commands']}/{report['command_coverage']['total_commands']}")
        
        if report['command_coverage']['missing_commands']:
            print(f"   ⚠️  Missing: {', '.join(report['command_coverage']['missing_commands'])}")
        
        print("\n📈 Top Commands:")
        for cmd, count in list(report['command_counts'].items())[:10]:
            pct = count / report['total_scenarios'] * 100
            bar = '█' * int(pct / 2)
            print(f"   {cmd:15s} {count:4d} ({pct:5.1f}%) {bar}")
        
        print("\n📁 Category Coverage:")
        for category, stats in report['category_coverage'].items():
            print(f"   {category:20s} {stats['percentage']:5.1f}% "
                  f"({stats['used']}/{stats['total']})")
            if stats['missing']:
                print(f"      Missing: {', '.join(stats['missing'])}")
        
        if report['underrepresented_commands']:
            threshold_pct = self.underrepresented_threshold * 100
            print(f"\n⚠️  Underrepresented Commands (< {threshold_pct:.0f}% of scenarios):")
            for cmd, count in sorted(report['underrepresented_commands'].items()):
                pct = count / report['total_scenarios'] * 100
                print(f"   {cmd}: {count} ({pct:.1f}%)")
        
        print("\n🎯 Scenario Types:")
        for stype, count in report['scenario_types'].items():
            pct = count / report['total_scenarios'] * 100
            print(f"   {stype:25s} {count:4d} ({pct:5.1f}%)")
        
        if report['recommendations']:
            print("\n💡 Recommendations:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"   {i}. {rec}")
        
        print("\n" + "="*80 + "\n")


def analyze_dataset_diversity(dataset_path: str) -> Dict[str, Any]:
    """Convenience function to analyze a dataset.
    
    Args:
        dataset_path: Path to dataset JSON file
        
    Returns:
        Diversity report dictionary
    """
    analyzer = DiversityAnalyzer()
    report = analyzer.analyze_dataset(dataset_path)
    analyzer.print_report(report)
    return report


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python diversity_analyzer.py <dataset_path>")
        sys.exit(1)
    
    analyze_dataset_diversity(sys.argv[1])

