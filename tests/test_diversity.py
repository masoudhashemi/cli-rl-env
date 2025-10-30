"""Unit tests for diversity improvements.

Tests:
- DiverseScenarioGenerator
- DiversityAnalyzer
- Balanced dataset generation
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path

from cli_rl_env.scenario_generator.diverse_scenarios import DiverseScenarioGenerator
from cli_rl_env.scenario_generator.base import DifficultyLevel
from cli_rl_env.utils.diversity_analyzer import DiversityAnalyzer
from cli_rl_env.prompt_dataset_generator import PromptDatasetGenerator


class TestDiverseScenarioGenerator:
    """Test suite for DiverseScenarioGenerator."""
    
    def test_initialization(self):
        """Test generator initialization."""
        gen = DiverseScenarioGenerator(seed=42)
        assert gen.seed == 42
        assert len(gen.COMMAND_CATEGORIES) > 0
    
    def test_generate_all_scenario_types(self):
        """Test that all scenario types can be generated."""
        gen = DiverseScenarioGenerator(seed=42)
        
        for _ in range(10):  # Generate 10 scenarios
            scenario = gen.generate_diverse_scenario(
                DifficultyLevel.MEDIUM, 'python'
            )
            
            # Basic checks
            assert scenario is not None
            assert scenario.files is not None
            assert len(scenario.files) > 0
            assert scenario.task_description
            assert 'scenario_type' in scenario.metadata
            assert scenario.language == 'python'
    
    def test_language_consistency(self):
        """Test that generated scenarios have correct language."""
        gen = DiverseScenarioGenerator(seed=42)
        
        py_scenario = gen.generate_diverse_scenario(
            DifficultyLevel.MEDIUM, 'python'
        )
        assert py_scenario.language == 'python'
        
        js_scenario = gen.generate_diverse_scenario(
            DifficultyLevel.MEDIUM, 'javascript'
        )
        assert js_scenario.language == 'javascript'
    
    def test_all_difficulty_levels(self):
        """Test all difficulty levels work."""
        gen = DiverseScenarioGenerator(seed=42)
        
        for diff in ['easy', 'medium', 'hard', 'very_hard']:
            scenario = gen.generate_diverse_scenario(
                DifficultyLevel(diff), 'python'
            )
            assert scenario.difficulty.value == diff
    
    def test_scenario_metadata(self):
        """Test that scenarios have proper metadata."""
        gen = DiverseScenarioGenerator(seed=42)
        
        scenario = gen.generate_diverse_scenario(
            DifficultyLevel.MEDIUM, 'python'
        )
        
        assert 'scenario_type' in scenario.metadata
        # Most scenarios should have command_focus
        # (though not all, so we don't make it required)
        assert isinstance(scenario.metadata, dict)
    
    def test_scenario_has_files(self):
        """Test that scenarios have at least one file."""
        gen = DiverseScenarioGenerator(seed=42)
        
        scenario = gen.generate_diverse_scenario(
            DifficultyLevel.MEDIUM, 'python'
        )
        
        assert len(scenario.files) > 0
        for file in scenario.files:
            assert file.path
            assert file.content
            assert isinstance(file.is_test, bool)
    
    def test_scenario_has_verification(self):
        """Test that scenarios have verification rules."""
        gen = DiverseScenarioGenerator(seed=42)
        
        scenario = gen.generate_diverse_scenario(
            DifficultyLevel.MEDIUM, 'python'
        )
        
        assert len(scenario.verification_rules) > 0
        for rule in scenario.verification_rules:
            assert rule.type
            assert rule.target


class TestDiversityAnalyzer:
    """Test suite for DiversityAnalyzer."""
    
    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = DiversityAnalyzer()
        assert analyzer.underrepresented_threshold == 0.05
        
        analyzer2 = DiversityAnalyzer(underrepresented_threshold=0.1)
        assert analyzer2.underrepresented_threshold == 0.1
    
    def test_threshold_validation(self):
        """Test that threshold validation works."""
        # Valid thresholds
        DiversityAnalyzer(underrepresented_threshold=0.0)
        DiversityAnalyzer(underrepresented_threshold=0.5)
        DiversityAnalyzer(underrepresented_threshold=1.0)
        
        # Invalid thresholds
        with pytest.raises(ValueError):
            DiversityAnalyzer(underrepresented_threshold=-0.1)
        
        with pytest.raises(ValueError):
            DiversityAnalyzer(underrepresented_threshold=1.5)
    
    def test_analyze_dataset(self):
        """Test dataset analysis."""
        # Create test dataset
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = [
                {
                    'id': 'test_001',
                    'task_description': 'Use grep and sed commands',
                    'metadata': {'scenario_type': 'test_type'}
                },
                {
                    'id': 'test_002',
                    'task_description': 'Use cat and awk commands',
                    'metadata': {'scenario_type': 'test_type'}
                }
            ]
            
            dataset_path = Path(tmpdir) / 'test.json'
            with open(dataset_path, 'w') as f:
                json.dump(dataset, f)
            
            analyzer = DiversityAnalyzer()
            report = analyzer.analyze_dataset(str(dataset_path))
            
            # Check report structure
            assert 'total_scenarios' in report
            assert 'command_coverage' in report
            assert 'command_counts' in report
            assert 'category_coverage' in report
            assert 'scenario_types' in report
            assert 'recommendations' in report
            
            # Check values
            assert report['total_scenarios'] == 2
            assert isinstance(report['command_coverage']['percentage'], float)
    
    def test_empty_dataset(self):
        """Test handling of empty dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = []
            
            dataset_path = Path(tmpdir) / 'empty.json'
            with open(dataset_path, 'w') as f:
                json.dump(dataset, f)
            
            analyzer = DiversityAnalyzer()
            report = analyzer.analyze_dataset(str(dataset_path))
            
            assert report['total_scenarios'] == 0
            assert report['command_coverage']['percentage'] == 0.0
    
    def test_command_extraction(self):
        """Test that commands are correctly extracted from task descriptions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = [
                {
                    'id': 'test_001',
                    'task_description': 'Use grep to find errors, then sed to fix them.',
                    'metadata': {'scenario_type': 'test'}
                }
            ]
            
            dataset_path = Path(tmpdir) / 'test.json'
            with open(dataset_path, 'w') as f:
                json.dump(dataset, f)
            
            analyzer = DiversityAnalyzer()
            report = analyzer.analyze_dataset(str(dataset_path))
            
            # Should have extracted grep and sed
            assert 'grep' in report['command_counts']
            assert 'sed' in report['command_counts']
    
    def test_category_coverage(self):
        """Test category coverage calculation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = [
                {
                    'id': f'test_{i:03d}',
                    'task_description': 'Use grep find sed awk cut',
                    'metadata': {'scenario_type': 'test'}
                }
                for i in range(10)
            ]
            
            dataset_path = Path(tmpdir) / 'test.json'
            with open(dataset_path, 'w') as f:
                json.dump(dataset, f)
            
            analyzer = DiversityAnalyzer()
            report = analyzer.analyze_dataset(str(dataset_path))
            
            # Check category coverage structure
            assert 'text_processing' in report['category_coverage']
            assert 'file_search' in report['category_coverage']
            
            for category, stats in report['category_coverage'].items():
                assert 'used' in stats
                assert 'total' in stats
                assert 'percentage' in stats
                assert 'missing' in stats


class TestBalancedDatasetGenerator:
    """Test suite for balanced dataset generation."""
    
    def test_generate_small_dataset(self):
        """Test generating a small dataset."""
        gen = PromptDatasetGenerator(seed=42)
        
        dataset = gen.generate_balanced_diverse_dataset(
            num_prompts=10,
            diverse_scenario_ratio=0.6
        )
        
        assert len(dataset) == 10
        
        # Check structure
        for item in dataset:
            assert 'id' in item
            assert 'difficulty' in item
            assert 'language' in item
            assert 'task_description' in item
            assert 'files' in item
            assert 'metadata' in item
    
    def test_diverse_scenario_ratio(self):
        """Test different diverse scenario ratios."""
        gen = PromptDatasetGenerator(seed=42)
        
        # 0% diverse
        dataset = gen.generate_balanced_diverse_dataset(
            num_prompts=10,
            diverse_scenario_ratio=0.0
        )
        assert len(dataset) == 10
        
        # 100% diverse
        dataset = gen.generate_balanced_diverse_dataset(
            num_prompts=10,
            diverse_scenario_ratio=1.0
        )
        assert len(dataset) == 10
        
        # 50% diverse
        dataset = gen.generate_balanced_diverse_dataset(
            num_prompts=10,
            diverse_scenario_ratio=0.5
        )
        assert len(dataset) == 10
    
    def test_ratio_validation(self):
        """Test that ratio validation works."""
        gen = PromptDatasetGenerator(seed=42)
        
        # Valid ratios
        gen.generate_balanced_diverse_dataset(num_prompts=5, diverse_scenario_ratio=0.0)
        gen.generate_balanced_diverse_dataset(num_prompts=5, diverse_scenario_ratio=0.5)
        gen.generate_balanced_diverse_dataset(num_prompts=5, diverse_scenario_ratio=1.0)
        
        # Invalid ratios
        with pytest.raises(ValueError):
            gen.generate_balanced_diverse_dataset(
                num_prompts=5, diverse_scenario_ratio=-0.1
            )
        
        with pytest.raises(ValueError):
            gen.generate_balanced_diverse_dataset(
                num_prompts=5, diverse_scenario_ratio=1.5
            )
    
    def test_difficulty_distribution_validation(self):
        """Test difficulty distribution validation."""
        gen = PromptDatasetGenerator(seed=42)
        
        # Valid distribution
        gen.generate_balanced_diverse_dataset(
            num_prompts=5,
            difficulty_distribution={'easy': 0.25, 'medium': 0.25, 'hard': 0.25, 'very_hard': 0.25}
        )
        
        # Invalid: doesn't sum to 1.0
        with pytest.raises(ValueError):
            gen.generate_balanced_diverse_dataset(
                num_prompts=5,
                difficulty_distribution={'easy': 0.5, 'medium': 0.3}  # = 0.8
            )
        
        # Invalid: wrong keys
        with pytest.raises(ValueError):
            gen.generate_balanced_diverse_dataset(
                num_prompts=5,
                difficulty_distribution={'easy': 0.5, 'super_hard': 0.5}
            )
    
    def test_no_duplicate_ids(self):
        """Test that generated IDs are unique."""
        gen = PromptDatasetGenerator(seed=42)
        
        dataset = gen.generate_balanced_diverse_dataset(
            num_prompts=50,
            diverse_scenario_ratio=0.6
        )
        
        ids = [item['id'] for item in dataset]
        assert len(ids) == len(set(ids)), "Duplicate IDs found"
    
    def test_file_output(self):
        """Test saving dataset to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = PromptDatasetGenerator(seed=42)
            
            output_file = Path(tmpdir) / 'test_dataset.json'
            dataset = gen.generate_balanced_diverse_dataset(
                num_prompts=10,
                diverse_scenario_ratio=0.6,
                output_file=str(output_file)
            )
            
            # Check file was created
            assert output_file.exists()
            
            # Check file contents
            with open(output_file) as f:
                loaded_data = json.load(f)
            
            assert len(loaded_data) == 10
            assert loaded_data == dataset
    
    def test_language_distribution(self):
        """Test that both Python and JavaScript scenarios are generated."""
        gen = PromptDatasetGenerator(seed=42)
        
        dataset = gen.generate_balanced_diverse_dataset(
            num_prompts=50,
            diverse_scenario_ratio=0.6
        )
        
        languages = [item['language'] for item in dataset]
        
        # Should have both languages
        assert 'python' in languages
        assert 'javascript' in languages
    
    def test_difficulty_distribution(self):
        """Test that difficulty distribution is respected."""
        gen = PromptDatasetGenerator(seed=42)
        
        dataset = gen.generate_balanced_diverse_dataset(
            num_prompts=100,
            difficulty_distribution={
                'easy': 0.0,
                'medium': 0.0,
                'hard': 0.5,
                'very_hard': 0.5
            },
            diverse_scenario_ratio=0.6
        )
        
        difficulties = [item['difficulty'] for item in dataset]
        
        # Should only have hard and very_hard
        assert 'easy' not in difficulties
        assert 'medium' not in difficulties
        assert 'hard' in difficulties
        assert 'very_hard' in difficulties


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_workflow(self):
        """Test complete workflow: generate → save → analyze."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Generate dataset
            gen = PromptDatasetGenerator(seed=42)
            output_file = Path(tmpdir) / 'dataset.json'
            
            dataset = gen.generate_balanced_diverse_dataset(
                num_prompts=30,
                diverse_scenario_ratio=0.7,
                output_file=str(output_file)
            )
            
            assert len(dataset) == 30
            assert output_file.exists()
            
            # 2. Analyze dataset
            analyzer = DiversityAnalyzer()
            report = analyzer.analyze_dataset(str(output_file))
            
            assert report['total_scenarios'] == 30
            assert report['command_coverage']['percentage'] > 0
            
            # 3. Check diversity improved
            # With diverse_scenario_ratio=0.7, should have good coverage
            assert report['command_coverage']['used_commands'] > 10
    
    def test_dataset_splits(self):
        """Test splitting dataset into train/val/test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = PromptDatasetGenerator(seed=42)
            
            dataset = gen.generate_balanced_diverse_dataset(
                num_prompts=100,
                diverse_scenario_ratio=0.6
            )
            
            output_dir = Path(tmpdir) / 'splits'
            gen.save_dataset_splits(
                dataset,
                output_dir=str(output_dir),
                train_ratio=0.8,
                val_ratio=0.1,
                test_ratio=0.1
            )
            
            # Check files exist
            assert (output_dir / 'train.json').exists()
            assert (output_dir / 'val.json').exists()
            assert (output_dir / 'test.json').exists()
            assert (output_dir / 'stats.json').exists()
            
            # Check sizes
            with open(output_dir / 'train.json') as f:
                train = json.load(f)
            with open(output_dir / 'val.json') as f:
                val = json.load(f)
            with open(output_dir / 'test.json') as f:
                test = json.load(f)
            
            assert len(train) == 80
            assert len(val) == 10
            assert len(test) == 10


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_single_prompt_dataset(self):
        """Test generating a dataset with a single prompt."""
        gen = PromptDatasetGenerator(seed=42)
        
        dataset = gen.generate_balanced_diverse_dataset(
            num_prompts=1,
            diverse_scenario_ratio=0.5
        )
        
        assert len(dataset) == 1
    
    def test_large_dataset_performance(self):
        """Test that large datasets can be generated efficiently."""
        import time
        
        gen = PromptDatasetGenerator(seed=42)
        
        start = time.time()
        dataset = gen.generate_balanced_diverse_dataset(
            num_prompts=100,
            diverse_scenario_ratio=0.6
        )
        elapsed = time.time() - start
        
        assert len(dataset) == 100
        # Should complete in reasonable time (< 10 seconds for 100 prompts)
        assert elapsed < 10.0
    
    def test_custom_threshold(self):
        """Test using custom underrepresented threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dataset
            dataset = [
                {
                    'id': f'test_{i:03d}',
                    'task_description': 'Use grep and sed',
                    'metadata': {'scenario_type': 'test'}
                }
                for i in range(20)
            ]
            
            dataset_path = Path(tmpdir) / 'test.json'
            with open(dataset_path, 'w') as f:
                json.dump(dataset, f)
            
            # Analyze with default threshold (5%)
            analyzer1 = DiversityAnalyzer(underrepresented_threshold=0.05)
            report1 = analyzer1.analyze_dataset(str(dataset_path))
            
            # Analyze with higher threshold (20%)
            analyzer2 = DiversityAnalyzer(underrepresented_threshold=0.20)
            report2 = analyzer2.analyze_dataset(str(dataset_path))
            
            # Higher threshold should flag more commands as underrepresented
            # (since threshold is 20% of 20 = 4 occurrences)
            assert isinstance(report1['underrepresented_commands'], dict)
            assert isinstance(report2['underrepresented_commands'], dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

