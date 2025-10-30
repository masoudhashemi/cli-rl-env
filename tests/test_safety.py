"""Tests for sandbox safety and cleanup."""

import os
import tempfile
import pytest
from pathlib import Path

from cli_rl_env.executor.sandbox import Sandbox
from cli_rl_env.scenario_generator.base import FileContent


def test_sandbox_cleanup():
    """Test that sandbox cleans up all files."""
    temp_path = None
    
    files = [FileContent(path="test.py", content="print('hello')", is_test=False)]
    
    with Sandbox(files) as sandbox:
        temp_path = sandbox.get_sandbox_path()
        assert os.path.exists(temp_path)
        
        # Execute command that creates additional files
        sandbox.execute_commands([
            "echo 'extra file' > extra.txt",
            "mkdir subdir",
            "echo 'nested' > subdir/nested.txt"
        ])
        
        # Verify files exist
        assert os.path.exists(os.path.join(temp_path, "test.py"))
        assert os.path.exists(os.path.join(temp_path, "extra.txt"))
        assert os.path.exists(os.path.join(temp_path, "subdir/nested.txt"))
    
    # After exiting context, everything should be cleaned up
    assert not os.path.exists(temp_path), "Sandbox directory should be deleted"


def test_sandbox_permissions():
    """Test that sandbox sets restrictive permissions."""
    files = [FileContent(path="test.py", content="print('hello')", is_test=False)]
    
    with Sandbox(files) as sandbox:
        sandbox_path = Path(sandbox.get_sandbox_path())
        
        # Check directory permissions (should be 700 = owner only)
        stat_info = os.stat(sandbox_path)
        mode = stat_info.st_mode & 0o777
        assert mode == 0o700, f"Expected 700, got {oct(mode)}"
        
        # Check file permissions (should be 600 = owner read/write only)
        test_file = sandbox_path / "test.py"
        stat_info = os.stat(test_file)
        mode = stat_info.st_mode & 0o777
        assert mode == 0o600, f"Expected 600, got {oct(mode)}"


def test_sandbox_path_traversal_prevention():
    """Test that path traversal attacks are prevented."""
    # Attempt to create file with path traversal
    files = [FileContent(path="../../../etc/malicious", content="bad", is_test=False)]
    
    with pytest.raises(ValueError, match="Path traversal"):
        with Sandbox(files) as sandbox:
            pass


def test_sandbox_navigation_restriction():
    """Test that cd cannot escape sandbox."""
    files = [FileContent(path="test.py", content="test", is_test=False)]
    
    with Sandbox(files) as sandbox:
        # Try to navigate outside sandbox
        result = sandbox.execute_commands([
            "cd ..",
            "cd ..",
            "cd ..",
            "pwd"
        ])
        
        # Should still be within sandbox
        pwd_output = result['results'][-1]['output'].strip()
        sandbox_path = sandbox.get_sandbox_path()
        assert pwd_output.startswith(sandbox_path), "Should not escape sandbox"


def test_sandbox_timeout():
    """Test that long-running commands are killed."""
    files = [FileContent(path="test.py", content="test", is_test=False)]
    
    with Sandbox(files, timeout=2) as sandbox:
        # This should timeout
        result = sandbox.execute_commands([
            "sleep 10"
        ])
        
        assert not result['all_successful']
        assert 'timed out' in result['results'][0]['error'].lower() or 'timeout' in result['results'][0]['error'].lower()


def test_sandbox_output_truncation():
    """Test that excessive output is truncated."""
    files = [FileContent(path="test.py", content="test", is_test=False)]
    
    with Sandbox(files) as sandbox:
        # Generate lots of output
        result = sandbox.execute_commands([
            "python3 -c \"print('x' * 200000)\""
        ])
        
        output = result['results'][0]['output']
        # Output should be truncated to prevent memory issues
        assert len(output) < 150000, "Output should be truncated"
        assert 'truncated' in output.lower() or len(output) < 150000


def test_sandbox_isolation():
    """Test that sandbox is isolated from system."""
    files = [FileContent(path="test.py", content="test", is_test=False)]
    
    with Sandbox(files) as sandbox:
        sandbox_path = sandbox.get_sandbox_path()
        
        # HOME should point to sandbox
        result = sandbox.execute_commands([
            "python3 -c \"import os; print(os.environ.get('HOME'))\""
        ])
        
        home = result['results'][0]['output'].strip()
        assert home == sandbox_path, "HOME should be isolated"


def test_multiple_sandboxes_independent():
    """Test that multiple sandboxes don't interfere."""
    files1 = [FileContent(path="test1.py", content="test1", is_test=False)]
    files2 = [FileContent(path="test2.py", content="test2", is_test=False)]
    
    with Sandbox(files1) as sandbox1:
        path1 = sandbox1.get_sandbox_path()
        
        with Sandbox(files2) as sandbox2:
            path2 = sandbox2.get_sandbox_path()
            
            # Should have different paths
            assert path1 != path2
            
            # Each should only have their own files
            assert os.path.exists(os.path.join(path1, "test1.py"))
            assert not os.path.exists(os.path.join(path1, "test2.py"))
            
            assert os.path.exists(os.path.join(path2, "test2.py"))
            assert not os.path.exists(os.path.join(path2, "test1.py"))


def test_sandbox_cleanup_on_exception():
    """Test that sandbox cleans up even when exception occurs."""
    files = [FileContent(path="test.py", content="test", is_test=False)]
    temp_path = None
    
    try:
        with Sandbox(files) as sandbox:
            temp_path = sandbox.get_sandbox_path()
            # Force an exception
            raise RuntimeError("Test exception")
    except RuntimeError:
        pass
    
    # Should still clean up
    assert not os.path.exists(temp_path), "Should clean up on exception"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

