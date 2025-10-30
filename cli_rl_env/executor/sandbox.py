"""Safe command execution in isolated sandbox environments."""

import os
import shutil
import subprocess
import tempfile
import time
import signal
from pathlib import Path
from typing import Dict, List, Any

from cli_rl_env.scenario_generator.base import FileContent


class Sandbox:
    """Isolated environment for executing commands with enhanced safety."""
    
    def __init__(self, files: List[FileContent], timeout: int = 30):
        """Initialize sandbox with files.
        
        Args:
            files: List of files to create in sandbox
            timeout: Maximum execution time per command in seconds
        """
        self.timeout = timeout
        self.temp_dir = None
        self.files = files
        self.current_dir = None
        self.execution_log = []
        self._original_dir = os.getcwd()
        
    def __enter__(self):
        """Set up the sandbox environment with safety measures."""
        # Create temporary directory with secure permissions
        self.temp_dir = tempfile.mkdtemp(prefix='cli_rl_env_', dir=tempfile.gettempdir())
        
        # Set restrictive permissions on temp directory (owner only)
        os.chmod(self.temp_dir, 0o700)
        
        self.current_dir = self.temp_dir
        
        # Write all files to sandbox with safe permissions
        for file_content in self.files:
            filepath = Path(self.temp_dir) / file_content.path
            
            # Prevent directory traversal attacks
            if not str(filepath.resolve()).startswith(str(Path(self.temp_dir).resolve())):
                raise ValueError(f"Security: Path traversal detected in {file_content.path}")
            
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(file_content.content)
            
            # Set safe permissions (read/write for owner only)
            os.chmod(filepath, 0o600)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up the sandbox completely and safely."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                # Force removal of all files, including any created during execution
                shutil.rmtree(self.temp_dir, ignore_errors=False)
            except Exception as e:
                # Log error but don't fail - try harder cleanup
                print(f"Warning: Initial cleanup failed: {e}")
                try:
                    # Try with more aggressive cleanup
                    for root, dirs, files in os.walk(self.temp_dir, topdown=False):
                        for name in files:
                            try:
                                os.chmod(os.path.join(root, name), 0o700)
                                os.remove(os.path.join(root, name))
                            except:
                                pass
                        for name in dirs:
                            try:
                                os.chmod(os.path.join(root, name), 0o700)
                                os.rmdir(os.path.join(root, name))
                            except:
                                pass
                    os.rmdir(self.temp_dir)
                except Exception as e2:
                    print(f"Warning: Aggressive cleanup also failed: {e2}")
        
        # Restore original directory
        try:
            os.chdir(self._original_dir)
        except:
            pass
    
    def execute_commands(self, commands: List[str]) -> Dict[str, Any]:
        """Execute a list of commands and measure time.
        
        Args:
            commands: List of command strings
            
        Returns:
            Dict with execution results and timing info
        """
        results = []
        start_time = time.time()
        
        for cmd in commands:
            cmd_start = time.time()
            try:
                result = self._execute_single_command(cmd)
                cmd_time = time.time() - cmd_start
                results.append({
                    'command': cmd,
                    'success': True,
                    'output': result,
                    'time': cmd_time
                })
                self.execution_log.append(f"$ {cmd}")
                self.execution_log.append(result[:500])  # Truncate long output
            except Exception as e:
                cmd_time = time.time() - cmd_start
                results.append({
                    'command': cmd,
                    'success': False,
                    'error': str(e),
                    'time': cmd_time
                })
                self.execution_log.append(f"$ {cmd}")
                self.execution_log.append(f"Error: {str(e)}")
        
        total_time = time.time() - start_time
        
        return {
            'results': results,
            'total_time': total_time,
            'all_successful': all(r['success'] for r in results)
        }
    
    def _execute_single_command(self, cmd: str) -> str:
        """Execute a single command with safety checks.
        
        Args:
            cmd: Command string
            
        Returns:
            Command output
            
        Raises:
            Exception: If command fails
        """
        parts = cmd.split()
        base_cmd = parts[0]
        
        # Handle shell built-ins that need special treatment
        if base_cmd == 'cd':
            return self._handle_cd(parts)
        elif base_cmd == 'pwd':
            return self.current_dir
        else:
            # Execute all commands via shell with safety
            return self._execute_shell_command(cmd)
    
    def _handle_cd(self, parts: List[str]) -> str:
        """Handle cd command safely."""
        if len(parts) < 2:
            self.current_dir = self.temp_dir
            return self.current_dir
        
        target = parts[1]
        if target == '..':
            parent = Path(self.current_dir).parent
            # Don't allow going above sandbox
            if str(parent).startswith(self.temp_dir):
                self.current_dir = str(parent)
            else:
                raise PermissionError("Cannot navigate outside sandbox")
        else:
            new_dir = Path(self.current_dir) / target
            # Verify we stay within sandbox
            if not str(new_dir.resolve()).startswith(str(Path(self.temp_dir).resolve())):
                raise PermissionError("Cannot navigate outside sandbox")
            
            if new_dir.exists() and new_dir.is_dir():
                self.current_dir = str(new_dir)
            else:
                raise FileNotFoundError(f"Directory not found: {target}")
        
        return self.current_dir
    
    def _execute_shell_command(self, cmd: str) -> str:
        """Execute a shell command safely with resource limits.
        
        Args:
            cmd: Command string
            
        Returns:
            Command output
        """
        try:
            # Create restricted environment
            safe_env = os.environ.copy()
            safe_env['PWD'] = self.current_dir
            safe_env['HOME'] = self.temp_dir  # Isolate home directory
            safe_env['TMPDIR'] = self.temp_dir  # Isolate temp directory
            
            # Remove potentially dangerous env vars
            for var in ['LD_PRELOAD', 'LD_LIBRARY_PATH', 'PYTHONPATH']:
                safe_env.pop(var, None)
            
            # Execute with timeout and security constraints
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=self.current_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=safe_env,
                # Additional safety: run with limited resources
                preexec_fn=self._limit_resources if os.name != 'nt' else None
            )
            
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR: {result.stderr}"
            
            # Truncate excessive output to prevent memory issues
            if len(output) > 100000:  # 100KB limit
                output = output[:100000] + f"\n... (truncated, {len(output)} total bytes)"
            
            if result.returncode != 0:
                raise RuntimeError(f"Command failed with code {result.returncode}: {output[:500]}")
            
            return output
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Command timed out after {self.timeout}s")
        except Exception as e:
            raise RuntimeError(f"Command execution failed: {str(e)}")
    
    def _limit_resources(self):
        """Limit resources for subprocess (Unix only)."""
        try:
            import resource
            
            # Limit CPU time (seconds)
            resource.setrlimit(resource.RLIMIT_CPU, (self.timeout, self.timeout))
            
            # Limit memory to 512MB
            mem_limit = 512 * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
            
            # Limit number of processes
            resource.setrlimit(resource.RLIMIT_NPROC, (50, 50))
            
            # Limit file size to 100MB
            file_size_limit = 100 * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_FSIZE, (file_size_limit, file_size_limit))
            
            # Limit number of open files
            resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
        except Exception:
            # Resource limits not available on this platform
            pass
    
    def get_file_contents(self) -> List[FileContent]:
        """Get current contents of all files in sandbox.
        
        Returns:
            List of FileContent objects with current state
        """
        result = []
        for file_content in self.files:
            filepath = Path(self.temp_dir) / file_content.path
            if filepath.exists():
                try:
                    content = filepath.read_text()
                    result.append(FileContent(
                        path=file_content.path,
                        content=content,
                        is_test=file_content.is_test
                    ))
                except Exception as e:
                    # File might have been deleted or is unreadable
                    print(f"Warning: Could not read {file_content.path}: {e}")
        return result
    
    def get_sandbox_path(self) -> str:
        """Get the sandbox temporary directory path.
        
        Returns:
            Path to sandbox directory
        """
        return self.temp_dir
