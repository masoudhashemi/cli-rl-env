"""Bug injection utilities for creating realistic debugging scenarios."""

import random
import re
from typing import List, Tuple, Optional


class BugInjector:
    """Injects realistic bugs into code."""
    
    @staticmethod
    def inject_python_bugs(code: str, num_bugs: int = 1) -> Tuple[str, List[str]]:
        """Inject bugs into Python code.
        
        Args:
            code: Original Python code
            num_bugs: Number of bugs to inject
            
        Returns:
            Tuple of (buggy_code, list of bug descriptions)
        """
        bugs_injected = []
        lines = code.split('\n')
        
        bug_types = [
            BugInjector._inject_syntax_error,
            BugInjector._inject_logic_error,
            BugInjector._inject_type_error,
            BugInjector._inject_missing_import,
            BugInjector._inject_wrong_operator,
        ]
        
        for _ in range(num_bugs):
            bug_func = random.choice(bug_types)
            lines, bug_desc = bug_func(lines, 'python')
            if bug_desc:
                bugs_injected.append(bug_desc)
        
        return '\n'.join(lines), bugs_injected
    
    @staticmethod
    def inject_javascript_bugs(code: str, num_bugs: int = 1) -> Tuple[str, List[str]]:
        """Inject bugs into JavaScript code.
        
        Args:
            code: Original JavaScript code
            num_bugs: Number of bugs to inject
            
        Returns:
            Tuple of (buggy_code, list of bug descriptions)
        """
        bugs_injected = []
        lines = code.split('\n')
        
        bug_types = [
            BugInjector._inject_syntax_error,
            BugInjector._inject_logic_error,
            BugInjector._inject_type_error,
            BugInjector._inject_wrong_operator,
        ]
        
        for _ in range(num_bugs):
            bug_func = random.choice(bug_types)
            lines, bug_desc = bug_func(lines, 'javascript')
            if bug_desc:
                bugs_injected.append(bug_desc)
        
        return '\n'.join(lines), bugs_injected
    
    @staticmethod
    def _inject_syntax_error(lines: List[str], lang: str) -> Tuple[List[str], str]:
        """Inject a syntax error."""
        # Find a line with a function definition or control structure
        for i, line in enumerate(lines):
            if lang == 'python' and ('def ' in line or 'class ' in line):
                if ':' in line:
                    lines[i] = line.replace(':', '')
                    return lines, f"Missing colon on line {i+1}"
            elif lang == 'javascript' and ('function' in line or 'const' in line):
                if '{' in line:
                    lines[i] = line.replace('{', '')
                    return lines, f"Missing opening brace on line {i+1}"
        return lines, ""
    
    @staticmethod
    def _inject_logic_error(lines: List[str], lang: str) -> Tuple[List[str], str]:
        """Inject a logic error."""
        # Find comparison or arithmetic operations
        for i, line in enumerate(lines):
            if '==' in line:
                lines[i] = line.replace('==', '!=')
                return lines, f"Wrong comparison operator on line {i+1}"
            elif ' > ' in line:
                lines[i] = line.replace(' > ', ' < ')
                return lines, f"Wrong comparison operator on line {i+1}"
        return lines, ""
    
    @staticmethod
    def _inject_type_error(lines: List[str], lang: str) -> Tuple[List[str], str]:
        """Inject a type-related error."""
        for i, line in enumerate(lines):
            if 'str(' in line or 'int(' in line:
                if 'str(' in line:
                    lines[i] = line.replace('str(', 'int(')
                    return lines, f"Wrong type conversion on line {i+1}"
                else:
                    lines[i] = line.replace('int(', 'str(')
                    return lines, f"Wrong type conversion on line {i+1}"
        return lines, ""
    
    @staticmethod
    def _inject_missing_import(lines: List[str], lang: str) -> Tuple[List[str], str]:
        """Remove an import statement."""
        if lang == 'python':
            for i, line in enumerate(lines):
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    lines[i] = '# ' + line
                    return lines, f"Commented out import on line {i+1}"
        return lines, ""
    
    @staticmethod
    def _inject_wrong_operator(lines: List[str], lang: str) -> Tuple[List[str], str]:
        """Change arithmetic operators."""
        for i, line in enumerate(lines):
            if ' + ' in line and 'return' in line:
                lines[i] = line.replace(' + ', ' - ')
                return lines, f"Wrong arithmetic operator on line {i+1}"
            elif ' * ' in line and 'return' in line:
                lines[i] = line.replace(' * ', ' / ')
                return lines, f"Wrong arithmetic operator on line {i+1}"
        return lines, ""

