"""Parse and validate LLM command outputs."""

import json
from typing import Dict, List, Any


class CommandParser:
    """Parse LLM outputs into executable commands."""
    
    # Whitelist of safe commands (standard CLI/bash tools only)
    SAFE_COMMANDS = {
        # File viewing
        'cat', 'head', 'tail', 'less', 'more',
        # File system
        'ls', 'find', 'tree', 'file', 'stat',
        # Text search/processing
        'grep', 'sed', 'awk', 'cut', 'tr', 'sort', 'uniq', 'wc',
        # File operations
        'cp', 'mv', 'rm', 'mkdir', 'touch', 'chmod',
        # Text editing
        'echo', 'printf', 'tee',
        # Comparison
        'diff', 'cmp', 'comm',
        # Patching
        'patch',
        # Navigation
        'cd', 'pwd',
        # Git
        'git',
        # Testing/execution
        'python', 'python3', 'node', 'pytest', 'npm',
        # Shell
        'bash', 'sh',
        # Other utilities
        'xargs', 'basename', 'dirname', 'which', 'type'
    }
    
    @staticmethod
    def parse_action(action: Any) -> Dict[str, Any]:
        """Parse action from LLM into structured format.
        
        Args:
            action: Action dict or JSON string with commands and time_estimate
            
        Returns:
            Dict with 'commands' list and 'time_estimate' float
            
        Raises:
            ValueError: If action format is invalid
        """
        if isinstance(action, str):
            try:
                action = json.loads(action)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in action: {e}")
        
        if not isinstance(action, dict):
            raise ValueError(f"Action must be dict or JSON string, got {type(action)}")
        
        if 'commands' not in action:
            raise ValueError("Action must contain 'commands' field")
        
        if 'time_estimate' not in action:
            raise ValueError("Action must contain 'time_estimate' field")
        
        commands = action['commands']
        if not isinstance(commands, list):
            raise ValueError("'commands' must be a list")
        
        time_estimate = action['time_estimate']
        if not isinstance(time_estimate, (int, float)):
            raise ValueError("'time_estimate' must be a number")
        
        # Validate each command
        validated_commands = []
        for cmd in commands:
            if not isinstance(cmd, str):
                raise ValueError(f"Command must be string, got {type(cmd)}")
            validated_cmd = CommandParser._validate_command(cmd)
            validated_commands.append(validated_cmd)
        
        return {
            'commands': validated_commands,
            'time_estimate': float(time_estimate)
        }
    
    @staticmethod
    def _validate_command(cmd: str) -> str:
        """Validate that command is safe to execute.
        
        Args:
            cmd: Command string to validate
            
        Returns:
            Validated command string
            
        Raises:
            ValueError: If command is unsafe
        """
        cmd = cmd.strip()
        
        if not cmd:
            raise ValueError("Empty command")
        
        # Extract base command (handle redirections and pipes)
        parts = cmd.split()
        base_cmd = parts[0]
        
        # Check if it's a safe command
        if base_cmd not in CommandParser.SAFE_COMMANDS:
            raise ValueError(f"Unsafe command: {base_cmd}. Allowed: {sorted(CommandParser.SAFE_COMMANDS)}")
        
        # Allow standard shell operators: >, >>, |, &&, ||
        # Disallow dangerous operators: ;, &(background), backticks, $()
        dangerous_patterns = [';', '`', '$(', '\n']
        for pattern in dangerous_patterns:
            if pattern in cmd:
                raise ValueError(f"Command contains unsafe operator: {pattern}")
        
        # Check for single & (background) but allow && (logical AND)
        if ' & ' in cmd or cmd.endswith('&'):
            raise ValueError("Background execution (&) not allowed")
        
        # Path traversal check
        if '..' in cmd:
            # Allow .. in some safe contexts (like cd ..)
            if base_cmd not in ['cd', 'ls', 'find']:
                raise ValueError(f"Command contains path traversal: {cmd}")
        
        # Disallow absolute paths and home directory (except for specific commands)
        parts_check = cmd.split()
        for part in parts_check:
            if part.startswith('/') and base_cmd not in ['find', 'grep', 'ls', 'git']:
                raise ValueError(f"Command uses absolute path: {part}")
            if '~' in part:
                raise ValueError(f"Command uses home directory: {part}")
        
        return cmd
