"""Verify text patterns in files."""

import re
from pathlib import Path
from typing import List, Dict, Any


class TextMatcher:
    """Match and verify text patterns in files."""
    
    def __init__(self, sandbox_path: str):
        """Initialize text matcher.
        
        Args:
            sandbox_path: Path to sandbox directory
        """
        self.sandbox_path = sandbox_path
    
    def verify_pattern(self, filepath: str, pattern: str, should_exist: bool = True) -> Dict[str, Any]:
        """Verify that a pattern exists or doesn't exist in a file.
        
        Args:
            filepath: Path to file (relative to sandbox)
            pattern: Pattern to search for (can be regex)
            should_exist: Whether pattern should exist (True) or not exist (False)
            
        Returns:
            Dict with verification results
        """
        full_path = Path(self.sandbox_path) / filepath
        
        if not full_path.exists():
            return {
                'success': False,
                'error': f'File not found: {filepath}'
            }
        
        try:
            content = full_path.read_text()
            
            # Try as regex first, fallback to literal search
            try:
                matches = re.findall(pattern, content)
                found = len(matches) > 0
            except re.error:
                # Not a valid regex, use literal search
                found = pattern in content
            
            success = found if should_exist else not found
            
            return {
                'success': success,
                'found': found,
                'should_exist': should_exist,
                'pattern': pattern,
                'filepath': filepath
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def verify_exact_match(self, filepath: str, expected_content: str) -> Dict[str, Any]:
        """Verify file content matches exactly.
        
        Args:
            filepath: Path to file
            expected_content: Expected file content
            
        Returns:
            Verification results
        """
        full_path = Path(self.sandbox_path) / filepath
        
        if not full_path.exists():
            return {
                'success': False,
                'error': f'File not found: {filepath}'
            }
        
        try:
            actual_content = full_path.read_text()
            success = actual_content == expected_content
            
            return {
                'success': success,
                'filepath': filepath,
                'matches': success
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def verify_line_count(self, filepath: str, expected_lines: int, tolerance: int = 0) -> Dict[str, Any]:
        """Verify file has expected number of lines.
        
        Args:
            filepath: Path to file
            expected_lines: Expected line count
            tolerance: Acceptable difference in line count
            
        Returns:
            Verification results
        """
        full_path = Path(self.sandbox_path) / filepath
        
        if not full_path.exists():
            return {
                'success': False,
                'error': f'File not found: {filepath}'
            }
        
        try:
            content = full_path.read_text()
            actual_lines = len(content.split('\n'))
            
            diff = abs(actual_lines - expected_lines)
            success = diff <= tolerance
            
            return {
                'success': success,
                'actual_lines': actual_lines,
                'expected_lines': expected_lines,
                'difference': diff
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

