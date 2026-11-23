import pytest
from app.security.path_guard import path_guard
from app.security.permissions import security_manager, ActionCategory
from app.tools.system_tools import calculate, get_system_info
from app.tools.registry import tool_registry

def test_path_guard_whitelist_and_traversal():
    # Valid path inside project data dir
    from app.config import DATA_DIR
    valid_file = str(DATA_DIR / "test.txt")
    assert path_guard.is_safe_path(valid_file) is True

    # Path traversal attack attempt
    traversal_path = str(DATA_DIR / ".." / ".." / "Windows" / "System32" / "cmd.exe")
    assert path_guard.is_safe_path(traversal_path) is False

    with pytest.raises(PermissionError):
        path_guard.validate_and_resolve(traversal_path)

def test_safe_calculate_tool():
    res1 = calculate("12 * 12")
    assert res1["status"] == "success"
    assert res1["result"] == 144

    res2 = calculate("100 / 4 + 25")
    assert res2["status"] == "success"
    assert res2["result"] == 50.0

    # Malicious code injection attempt should fail safely
    res3 = calculate("__import__('os').system('dir')")
    assert res3["status"] == "error"

def test_system_info_tool():
    info = get_system_info()
    assert "os" in info
    assert "disk_c" in info
    assert info["disk_c"]["free_gb"] > 0

@pytest.mark.asyncio
async def test_tool_authorization_barrier():
    # Attempting to execute write_file without prior authorization
    result = await tool_registry.execute_tool(
        name="write_file",
        params={"path": "test.txt", "content": "hello"},
        authorized=False
    )
    assert result["status"] == "requires_authorization"
    assert "action_id" in result
    assert result["category"] == ActionCategory.REQUIRES_AUTH.value
