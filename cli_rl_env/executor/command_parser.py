"""Parse and validate LLM command outputs."""

import json
import platform
import re
from typing import Dict, List, Any


class CommandParser:
    """Parse LLM outputs into executable commands."""
    
    # Whitelist of safe commands (standard CLI/bash tools only)
    SAFE_COMMANDS = {
        # File viewing
        'cat', 'head', 'tail', 'less', 'more', 'tac',
        # File system
        'ls', 'find', 'tree', 'file', 'stat', 'du', 'df',
        # Text search/processing
        'grep', 'sed', 'awk', 'cut', 'tr', 'sort', 'uniq', 'wc', 'nl', 'paste', 'column', 'expand', 'unexpand',
        # File operations
        'cp', 'mv', 'rm', 'mkdir', 'touch', 'chmod', 'chown', 'ln', 'readlink', 'realpath',
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
        'python', 'python3', 'node', 'pytest', 'npm', 'jest', 'mocha', 'pip', 'pip3',
        # Shell
        'bash', 'sh',
        # Data generation/manipulation
        'seq', 'yes', 'true', 'false', 'shuf',
        # System info (read-only)
        'date', 'env', 'printenv', 'whoami', 'hostname', 'uname', 'uptime',
        # Compression (read-only operations are safe)
        'tar', 'gzip', 'gunzip', 'zip', 'unzip', 'bzip2', 'bunzip2',
        # Checksums (safe, read-only)
        'md5sum', 'sha1sum', 'sha256sum', 'cksum',
        # Other utilities
        'xargs', 'basename', 'dirname', 'which', 'type', 'test', '[', 'expr', 'bc', 'jq'
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
    def _normalize_sed_command(cmd: str) -> str:
        """Normalize sed commands for cross-platform compatibility.
        
        Handles differences between BSD sed (macOS) and GNU sed (Linux).
        Main differences:
        - macOS requires an argument after -i flag (use -i '' for no backup)
        - Insert/append/change commands need different handling
        
        Args:
            cmd: sed command string
            
        Returns:
            Normalized sed command string
        """
        
        is_macos = platform.system().lower() == 'darwin'
        
        # Handle -i flag (in-place editing)
        # On macOS, -i requires an argument (backup extension)
        # GNU sed accepts -i without argument
        if is_macos:
            # Check if command has -i without an empty string argument already
            # Match: sed -i 's/...  (not sed -i '' 's/... or sed -i "" 's/...)
            # We need to add '' after -i if it's not already there
            if re.search(r"sed\s+-i\s+(?!['\"]['\"]\s)", cmd):
                # Insert empty string argument for macOS
                cmd = re.sub(r'(sed\s+-i)(\s+)', r"\1 '' ", cmd)
        else:
            # On Linux, remove empty string argument if present
            # Match: sed -i '' or sed -i ""
            cmd = re.sub(r"sed\s+-i\s+['\"]['\"]\s+", "sed -i ", cmd)
        
        # Handle insert/append/change commands (i/a/c)
        # These work on both platforms with similar syntax
        # Light normalization for common patterns, otherwise keep model's format unchanged
        
        # Pattern: sed 'NAi\text' or sed 'NAa\text' or sed 'NAc\text' where N is line number/pattern
        # Both BSD and GNU sed support these, just need proper formatting
        
        # Ensure backslash-newline sequences are properly formatted for multi-line text
        # Common pattern: 'i\<newline>text' should work on both platforms
        # Model should generate correct platform-specific format; we just validate basic structure
        
        # Check for common i/a/c patterns and ensure they have proper structure
        if re.search(r'[iac]\\', cmd):
            # Has insert/append/change with backslash - this is acceptable on both platforms
            # The model should have generated the right format for the target platform
            pass
        
        return cmd
    
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
        # Disallow dangerous operators: ;, &(background), backticks, $(), heredocs, multi-line embeddings
        # Note: Semicolons are allowed in sed/awk patterns (they appear in text being edited)
        dangerous_patterns = ['`', '$(', '\n', '<<', '<<-']
        for pattern in dangerous_patterns:
            if pattern in cmd:
                raise ValueError(f"Command contains unsafe operator: {pattern}")
        
        # Check for semicolon command chaining (but allow in sed/awk expressions)
        # Dangerous: cmd1 ; cmd2 or cmd1; cmd2
        # Safe: sed 's/old;/new;/' or awk pattern with semicolons
        if ';' in cmd:
            # Allow semicolons in sed and awk commands (common in patterns)
            if base_cmd not in ['sed', 'awk']:
                raise ValueError(f"Command contains unsafe operator: ; (semicolon command chaining not allowed)")
        
        # Check for single & (background) but allow && (logical AND)
        if ' & ' in cmd or cmd.endswith('&'):
            raise ValueError("Background execution (&) not allowed")

        # Block common multi-line embedding patterns (e.g., python - <<'PY')
        if ' - <<' in cmd or '<<EOF' in cmd or '<<' in cmd:
            raise ValueError("Heredocs or multi-line embeddings are not allowed")
        
        # Normalize sed commands for cross-platform compatibility (macOS/Linux)
        if base_cmd == 'sed':
            cmd = CommandParser._normalize_sed_command(cmd)
        
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
