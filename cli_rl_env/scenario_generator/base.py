"""Base classes for scenario generation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any


class DifficultyLevel(Enum):
    """Difficulty levels for scenarios."""
    EASY = "easy"  # 1-2 commands
    MEDIUM = "medium"  # 3-5 commands
    HARD = "hard"  # 6-10 commands
    VERY_HARD = "very_hard"  # 10+ commands


@dataclass
class FileContent:
    """Represents a file in the scenario."""
    path: str
    content: str
    is_test: bool = False


@dataclass
class VerificationRule:
    """Rules for verifying task completion."""
    type: str  # 'test', 'text_match', 'lint', 'execution'
    target: Optional[str] = None  # file path or test name
    expected: Optional[Any] = None  # expected value/pattern
    description: str = ""


@dataclass
class Scenario:
    """Complete scenario for an RL episode."""
    difficulty: DifficultyLevel
    language: str  # 'python' or 'javascript'
    task_description: str
    files: List[FileContent]
    verification_rules: List[VerificationRule]
    expected_commands: int  # Expected number of commands to solve
    cli_history: List[str]  # Simulated CLI history
    solution: Optional[List[str]] = None  # Optional reference solution
    metadata: Optional[Dict[str, Any]] = None


class ScenarioGenerator(ABC):
    """Abstract base class for scenario generators."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize the generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            import random
            random.seed(seed)
    
    @abstractmethod
    def generate(self, difficulty: DifficultyLevel) -> Scenario:
        """Generate a scenario of the specified difficulty.
        
        Args:
            difficulty: Target difficulty level
            
        Returns:
            Complete scenario
        """
        pass
    
    @abstractmethod
    def get_language(self) -> str:
        """Get the programming language for this generator.
        
        Returns:
            Language name ('python' or 'javascript')
        """
        pass

