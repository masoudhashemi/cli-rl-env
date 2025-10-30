"""Simulate CLI history for scenarios."""

from typing import List
from cli_rl_env.scenario_generator.base import FileContent


class CLISimulator:
    """Generate realistic CLI history for scenarios."""
    
    @staticmethod
    def generate_initial_history(files: List[FileContent], include_errors: bool = False) -> List[str]:
        """Generate initial CLI history showing file exploration.
        
        Args:
            files: Files in the scenario
            include_errors: Whether to include error messages
            
        Returns:
            List of CLI history lines
        """
        history = []
        
        # List files
        history.append("$ ls")
        history.append("  ".join([f.path for f in files]))
        history.append("")
        
        # Show file structure
        history.append("$ ls -lh")
        for f in files:
            size = len(f.content)
            history.append(f"-rw-r--r-- 1 user user {size:>6} Oct 30 10:00 {f.path}")
        history.append("")
        
        if include_errors:
            # Simulate test run with errors
            test_files = [f for f in files if f.is_test]
            if test_files:
                history.append(f"$ pytest {test_files[0].path} -v")
                history.append("============================= test session starts ==============================")
                history.append("collected 3 items")
                history.append("")
                history.append(f"{test_files[0].path}::test_function FAILED")
                history.append("")
                history.append("=================================== FAILURES ===================================")
                history.append("Some tests are failing. Debug and fix the code.")
                history.append("")
        
        return history
    
    @staticmethod
    def format_command_output(command: str, output: str, max_lines: int = 20) -> List[str]:
        """Format command and its output for history.
        
        Args:
            command: Command that was run
            output: Output from the command
            max_lines: Maximum lines to include from output
            
        Returns:
            Formatted history lines
        """
        history = [f"$ {command}"]
        
        lines = output.split('\n')
        if len(lines) > max_lines:
            history.extend(lines[:max_lines])
            history.append(f"... ({len(lines) - max_lines} more lines)")
        else:
            history.extend(lines)
        
        history.append("")
        return history

