"""Run unit tests to verify code correctness."""

import subprocess
import os
from typing import Dict, Any, List
from pathlib import Path


class TestRunner:
    """Run and evaluate unit tests."""
    
    def __init__(self, sandbox_path: str, language: str):
        """Initialize test runner.
        
        Args:
            sandbox_path: Path to sandbox directory
            language: Programming language ('python' or 'javascript')
        """
        self.sandbox_path = sandbox_path
        self.language = language
    
    def run_tests(self, test_file: str, timeout: int = 30) -> Dict[str, Any]:
        """Run tests and return results.
        
        Args:
            test_file: Path to test file (relative to sandbox)
            timeout: Maximum time for test execution
            
        Returns:
            Dict with test results
        """
        if self.language == 'python':
            return self._run_python_tests(test_file, timeout)
        elif self.language == 'javascript':
            return self._run_javascript_tests(test_file, timeout)
        else:
            raise ValueError(f"Unsupported language: {self.language}")
    
    def _run_python_tests(self, test_file: str, timeout: int) -> Dict[str, Any]:
        """Run Python tests using pytest.
        
        Args:
            test_file: Test file path
            timeout: Timeout in seconds
            
        Returns:
            Test results dict
        """
        try:
            result = subprocess.run(
                ['pytest', test_file, '-v', '--tb=short', '--timeout=10'],
                cwd=self.sandbox_path,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
            )
            
            output = result.stdout + result.stderr
            
            # Parse pytest output
            passed = self._count_pytest_passed(output)
            failed = self._count_pytest_failed(output)
            total = passed + failed
            
            return {
                'success': result.returncode == 0,
                'passed': passed,
                'failed': failed,
                'total': total,
                'output': output,
                'exit_code': result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'passed': 0,
                'failed': 0,
                'total': 0,
                'output': 'Tests timed out',
                'exit_code': -1,
                'error': 'timeout'
            }
        except Exception as e:
            return {
                'success': False,
                'passed': 0,
                'failed': 0,
                'total': 0,
                'output': str(e),
                'exit_code': -1,
                'error': str(e)
            }
    
    def _run_javascript_tests(self, test_file: str, timeout: int) -> Dict[str, Any]:
        """Run JavaScript tests using node.
        
        Args:
            test_file: Test file path
            timeout: Timeout in seconds
            
        Returns:
            Test results dict
        """
        try:
            result = subprocess.run(
                ['node', test_file],
                cwd=self.sandbox_path,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            output = result.stdout + result.stderr
            
            # Count passed tests (look for ✓ or "passed" messages)
            passed = output.count('✓') or output.count('passed')
            
            return {
                'success': result.returncode == 0,
                'passed': passed if result.returncode == 0 else 0,
                'failed': 0 if result.returncode == 0 else 1,
                'total': passed if result.returncode == 0 else 1,
                'output': output,
                'exit_code': result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'passed': 0,
                'failed': 1,
                'total': 1,
                'output': 'Tests timed out',
                'exit_code': -1,
                'error': 'timeout'
            }
        except Exception as e:
            return {
                'success': False,
                'passed': 0,
                'failed': 1,
                'total': 1,
                'output': str(e),
                'exit_code': -1,
                'error': str(e)
            }
    
    @staticmethod
    def _count_pytest_passed(output: str) -> int:
        """Count passed tests in pytest output."""
        # Look for "X passed" in output
        import re
        match = re.search(r'(\d+) passed', output)
        if match:
            return int(match.group(1))
        return 0
    
    @staticmethod
    def _count_pytest_failed(output: str) -> int:
        """Count failed tests in pytest output."""
        import re
        match = re.search(r'(\d+) failed', output)
        if match:
            return int(match.group(1))
        return 0

