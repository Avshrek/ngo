"""End-to-end HTTP integration test for N-GO server."""
import httpx
import json
import os

BASE = "http://localhost:8000"
client = httpx.Client(timeout=30.0)

def step(tool_name, arguments=None):
    """Helper: call /step with correct OpenEnv format."""
    r = client.post(f"{BASE}/step", json={
        "action": {"tool_name": tool_name, "arguments": arguments or {}}
    })
    assert r.status_code == 200, f"/step {tool_name} failed: {r.status_code} — {r.text[:200]}"
    return r.json()

print("=" * 60)
print("  N-GO HTTP Integration Test")
print("=" * 60)

# 1. Reset - cost_hemorrhage
print("\n[1] POST /reset (cost_hemorrhage)")
r = client.post(f"{BASE}/reset", json={"task_name": "cost_hemorrhage"})
assert r.status_code == 200
print(f"  ✓ Status 200")

# 2. get_telemetry
print("[2] get_telemetry()")
data = step("get_telemetry")
print(f"  ✓ Status 200")

# 3. update_routing_rule
print("[3] update_routing_rule(complexity_score lt 0.3 → 8B)")
data = step("update_routing_rule", {
    "field": "complexity_score", "operator": "lt", "value": "0.3",
    "target_model": "llama-3-8b-instruct", "priority": 10
})
print(f"  ✓ Status 200")

# 4. grade_config (cost)
print("[4] grade_config() — cost task")
data = step("grade_config")
print(f"  ✓ Status 200")

# 5. Reset - pii_leak
print("\n[5] POST /reset (pii_leak)")
r = client.post(f"{BASE}/reset", json={"task_name": "pii_leak"})
assert r.status_code == 200
print(f"  ✓ Status 200")

# 6. deploy_middleware (SSN)
print("[6] deploy_middleware(SSN)")
data = step("deploy_middleware", {
    "pattern": "\\b\\d{3}-\\d{2}-\\d{4}\\b",
    "replacement": "[REDACTED-SSN]",
    "name": "ssn_scrubber"
})
print(f"  ✓ Status 200")

# 7. deploy_middleware (CC)
print("[7] deploy_middleware(CC)")
data = step("deploy_middleware", {
    "pattern": "\\b\\d{4}[-\\s]?\\d{4}[-\\s]?\\d{4}[-\\s]?\\d{4}\\b",
    "replacement": "[REDACTED-CC]",
    "name": "cc_scrubber"
})
print(f"  ✓ Status 200")

# 8. grade_config (pii)
print("[8] grade_config() — PII task")
data = step("grade_config")
print(f"  ✓ Status 200")

# 9. Reset - jailbreak_cascade
print("\n[9] POST /reset (jailbreak_cascade)")
r = client.post(f"{BASE}/reset", json={"task_name": "jailbreak_cascade"})
assert r.status_code == 200
print(f"  ✓ Status 200")

# 10. block_ip
print("[10] block_ip(192.168.1.100)")
data = step("block_ip", {"ip_address": "192.168.1.100"})
print(f"  ✓ Status 200")

# 11. inject_system_prompt
print("[11] inject_system_prompt(nemotron-3)")
data = step("inject_system_prompt", {
    "model_name": "nemotron-3",
    "prompt": "REJECT REFUSE BLOCK DENY reject_encoded base64 rot13"
})
print(f"  ✓ Status 200")

# 12. grade_config (jailbreak)
print("[12] grade_config() — jailbreak task")
data = step("grade_config")
print(f"  ✓ Status 200")

print("\n" + "=" * 60)
print("  ✅ ALL 12 HTTP TESTS PASSED!")
print("  All 3 tasks: reset → tools → grade_config flow verified")
print("=" * 60)

os._exit(0)
