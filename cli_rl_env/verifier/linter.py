"""Run linting tools to verify code quality."""

import subprocess
import os
from typing import Dict, Any
from pathlib import Path


class Linter:
    """Run linting tools on code."""
    
    def __init__(self, sandbox_path: str, language: str):
        """Initialize linter.
        
        Args:
            sandbox_path: Path to sandbox directory
            language: Programming language ('python' or 'javascript')
        """
        self.sandbox_path = sandbox_path
        self.language = language
    
    def lint_file(self, filepath: str, strict: bool = False) -> Dict[str, Any]:
        """Lint a file and return results.
        
        Args:
            filepath: Path to file (relative to sandbox)
            strict: Whether to use strict linting rules
            
        Returns:
            Linting results dict
        """
        if self.language == 'python':
            return self._lint_python(filepath, strict)
        elif self.language == 'javascript':
            return self._lint_javascript(filepath, strict)
        else:
            raise ValueError(f"Unsupported language: {self.language}")
    
    def _lint_python(self, filepath: str, strict: bool) -> Dict[str, Any]:
        """Lint Python file using flake8.
        
        Args:
            filepath: File to lint
            strict: Whether to use strict rules
            
        Returns:
            Linting results
        """
        full_path = Path(self.sandbox_path) / filepath
        
        if not full_path.exists():
            return {
                'success': False,
                'error': f'File not found: {filepath}'
            }
        
        try:
            # Use flake8 with reasonable defaults
            args = ['flake8', filepath, '--max-line-length=100']
            
            if not strict:
                # Ignore some common issues for non-strict mode
                args.extend(['--ignore=E501,W503,E203'])
            
            result = subprocess.run(
                args,
                cwd=self.sandbox_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            output = result.stdout + result.stderr
            
            # Count errors and warnings
            error_count = len([line for line in output.split('\n') if line.strip()])
            
            return {
                'success': result.returncode == 0,
                'error_count': error_count,
                'output': output,
                'exit_code': result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Linting timed out',
                'exit_code': -1
            }
        except FileNotFoundError:
            # flake8 not installed, skip linting
            return {
                'success': True,
                'skipped': True,
                'message': 'flake8 not installed, skipping linting',
                'error_count': 0
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'exit_code': -1
            }
    
    def _lint_javascript(self, filepath: str, strict: bool) -> Dict[str, Any]:
        """Lint JavaScript file.
        
        Args:
            filepath: File to lint
            strict: Whether to use strict rules
            
        Returns:
            Linting results
        """
        full_path = Path(self.sandbox_path) / filepath
        
        if not full_path.exists():
            return {
                'success': False,
                'error': f'File not found: {filepath}'
            }
        
        # Basic syntax check using node
        try:
            result = subprocess.run(
                ['node', '--check', filepath],
                cwd=self.sandbox_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            output = result.stdout + result.stderr
            
            return {
                'success': result.returncode == 0,
                'output': output,
                'exit_code': result.returncode,
                'error_count': 0 if result.returncode == 0 else 1
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Linting timed out',
                'exit_code': -1
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'exit_code': -1
            }

