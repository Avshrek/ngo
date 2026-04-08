"""
N-GO Test Suites — Hardcoded deterministic test suites for all 3 tasks.

Each suite has exactly 1000 requests. The grader runs the agent's final
GatewayConfig against these suites and returns a score from 0.0 to 1.0.
"""

import re
from typing import Dict, List, Tuple

from .gateway_engine import (
    GatewayConfig,
    GatewayEngine,
    GatewayRequest,
    RequestResult,
)
from .traffic_generator import (
    generate_attack_requests,
    generate_benign_complex,
    generate_benign_simple,
    generate_clean_with_account_ids,
    generate_normal_traffic,
    generate_pii_requests,
)


# ---------------------------------------------------------------------------
# Task 1: Cost Hemorrhage — Easy
# ---------------------------------------------------------------------------

def build_cost_test_suite() -> List[GatewayRequest]:
    """
    1000 requests: 500 simple (should go to 8B) + 500 complex (should go to 70B+).

    The broken default routes EVERYTHING to nemotron-3 (most expensive).
    Agent must add routing rules to divert simple tasks to llama-3-8b-instruct.
    """
    simple = generate_benign_simple(500, seed=1001)
    complex_ = generate_benign_complex(500, seed=1002)
    return simple + complex_


def grade_cost_hemorrhage(config: GatewayConfig) -> Tuple[float, Dict]:
    """
    Grade the agent's config for the Cost Hemorrhage task.

    Scoring:
      - Calculate cost with broken defaults (all to nemotron-3)
      - Calculate cost with agent's config
      - Score = 0.5 * (has_routing_rule) + 0.5 * min(1.0, cost_reduction / 0.4)

    Returns: (score, details_dict)
    """
    suite = build_cost_test_suite()

    # Baseline cost: everything goes to nemotron-3
    baseline_config = GatewayConfig(default_model="nemotron-3")
    baseline_engine = GatewayEngine(baseline_config)
    baseline_engine.process_batch(suite)
    baseline_cost = baseline_engine.total_cost

    # Agent's config cost
    agent_engine = GatewayEngine(config)
    agent_engine.process_batch(suite)
    agent_cost = agent_engine.total_cost

    # Score components
    has_routing_rules = 0.5 if len(config.routing_rules) > 0 else 0.0

    cost_reduction = (baseline_cost - agent_cost) / baseline_cost if baseline_cost > 0 else 0.0
    target_reduction = 0.4  # 40% reduction target

    cost_score = 0.5 * min(1.0, max(0.0, cost_reduction / target_reduction))

    total_score = has_routing_rules + cost_score

    details = {
        "baseline_cost_usd": round(baseline_cost, 4),
        "agent_cost_usd": round(agent_cost, 4),
        "cost_reduction_pct": round(cost_reduction * 100, 2),
        "target_reduction_pct": target_reduction * 100,
        "has_routing_rules": has_routing_rules > 0,
        "routing_rule_count": len(config.routing_rules),
        "score_breakdown": {
            "routing_identification": has_routing_rules,
            "cost_optimization": round(cost_score, 4),
        },
    }

    return round(min(0.999, max(0.001, total_score)), 4), details



# ---------------------------------------------------------------------------
# Task 2: PII Leak — Medium
# ---------------------------------------------------------------------------

def build_pii_test_suite() -> Tuple[List[GatewayRequest], List[GatewayRequest]]:
    """
    1000 requests: 500 PII-containing + 250 clean + 250 with account IDs.

    The 250 account ID requests contain valid 9-digit numbers that should
    NOT be redacted (false positive trap).
    """
    pii_requests = generate_pii_requests(500, seed=2001)
    clean_requests = generate_benign_simple(250, seed=2002)
    account_requests = generate_clean_with_account_ids(250, seed=2003)
    return pii_requests, clean_requests + account_requests


def grade_pii_leak(config: GatewayConfig) -> Tuple[float, Dict]:
    """
    Grade the agent's config for the PII Leak task.

    Scoring:
      - +0.3 for SSN catch rate
      - +0.3 for Credit Card catch rate
      - +0.4 for avoiding false positives on account IDs/clean text

    Returns: (score, details_dict)
    """
    pii_requests, clean_requests = build_pii_test_suite()
    engine = GatewayEngine(config)

    # Process PII requests and check if PII was scrubbed
    ssn_pattern = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
    cc_pattern = re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b|\b\d{13,16}\b')

    ssn_total = 0
    ssn_caught = 0
    cc_total = 0
    cc_caught = 0

    for req in pii_requests:
        result = engine.process_request(req)
        payload = result.scrubbed_payload or req.prompt

        # Count SSNs in original
        original_ssns = ssn_pattern.findall(req.prompt)
        remaining_ssns = ssn_pattern.findall(payload)
        ssn_total += len(original_ssns)
        ssn_caught += len(original_ssns) - len(remaining_ssns)

        # Count CCs in original
        original_ccs = cc_pattern.findall(req.prompt)
        remaining_ccs = cc_pattern.findall(payload)
        cc_total += len(original_ccs)
        cc_caught += len(original_ccs) - len(remaining_ccs)

    # Process clean requests and check for false positives
    false_positives = 0
    clean_total = len(clean_requests)

    for req in clean_requests:
        result = engine.process_request(req)
        if result.scrubbed_payload and result.scrubbed_payload != req.prompt:
            # Something was redacted from a clean request = false positive
            false_positives += 1

    # Score components
    ssn_rate = ssn_caught / ssn_total if ssn_total > 0 else 0.0
    cc_rate = cc_caught / cc_total if cc_total > 0 else 0.0
    fp_rate = false_positives / clean_total if clean_total > 0 else 0.0

    ssn_score = 0.3 * ssn_rate
    cc_score = 0.3 * cc_rate
    fp_score = 0.4 * (1.0 - fp_rate)

    total_score = ssn_score + cc_score + fp_score

    details = {
        "ssn_total": ssn_total,
        "ssn_caught": ssn_caught,
        "ssn_rate": round(ssn_rate, 4),
        "cc_total": cc_total,
        "cc_caught": cc_caught,
        "cc_rate": round(cc_rate, 4),
        "clean_total": clean_total,
        "false_positives": false_positives,
        "false_positive_rate": round(fp_rate, 4),
        "score_breakdown": {
            "ssn_scrubbing": round(ssn_score, 4),
            "cc_scrubbing": round(cc_score, 4),
            "false_positive_avoidance": round(fp_score, 4),
        },
    }

    return round(min(0.999, max(0.001, total_score)), 4), details



# ---------------------------------------------------------------------------
# Task 3: Jailbreak Cascade — Hard
# ---------------------------------------------------------------------------

def build_jailbreak_test_suite() -> Tuple[List[GatewayRequest], List[GatewayRequest]]:
    """
    1000 requests: 500 attack + 500 normal.

    Attack vectors: Base64, ROT13, direct jailbreaks, roleplay.
    """
    attacks = generate_attack_requests(500, seed=3001)
    normal = generate_normal_traffic(500, seed=3002)
    return attacks, normal


def grade_jailbreak_cascade(config: GatewayConfig) -> Tuple[float, Dict]:
    """
    Grade the agent's config for the Jailbreak Cascade task.

    Scoring:
      - +0.2 for blocking attacker IPs
      - +0.3 for blocking Base64/ROT13 encoded payloads
      - +0.5 for blocking all jailbreaks while allowing normal traffic through

    Returns: (score, details_dict)
    """
    attacks, normal = build_jailbreak_test_suite()

    from .traffic_generator import ATTACKER_IPS

    # Process all attacks in a single pass, store results
    attack_engine = GatewayEngine(config)
    attack_results = []
    for req in attacks:
        result = attack_engine.process_request(req)
        attack_results.append((req, result))

    # Analyze attack results
    total_attacks = len(attacks)
    blocked_attacks = sum(1 for _, r in attack_results if r.blocked)

    # Count encoded attacks that were blocked
    encoded_total = sum(1 for req, _ in attack_results if req.attack_type in ("base64", "rot13"))
    encoded_caught = sum(
        1 for req, r in attack_results
        if req.attack_type in ("base64", "rot13") and r.blocked
    )
    encoded_block_rate = encoded_caught / encoded_total if encoded_total > 0 else 0.0

    # Count how many attacker IPs are in the blocklist
    attacker_ips_blocked = sum(1 for ip in ATTACKER_IPS if ip in config.blocked_ips)
    ip_block_rate = attacker_ips_blocked / len(ATTACKER_IPS)

    # Process normal traffic (should NOT be blocked) — fresh engine
    normal_engine = GatewayEngine(config)
    normal_total = len(normal)
    normal_blocked = 0
    for req in normal:
        result = normal_engine.process_request(req)
        if result.blocked:
            normal_blocked += 1

    normal_pass_rate = (normal_total - normal_blocked) / normal_total if normal_total > 0 else 0.0
    jailbreak_block_rate = blocked_attacks / total_attacks if total_attacks > 0 else 0.0

    # Score components
    ip_score = 0.2 * ip_block_rate
    encoded_score = 0.3 * encoded_block_rate
    defense_score = 0.5 * (jailbreak_block_rate * normal_pass_rate)

    total_score = ip_score + encoded_score + defense_score

    details = {
        "total_attacks": total_attacks,
        "blocked_attacks": blocked_attacks,
        "jailbreak_block_rate": round(jailbreak_block_rate, 4),
        "attacker_ips_blocked": attacker_ips_blocked,
        "attacker_ips_total": len(ATTACKER_IPS),
        "ip_block_rate": round(ip_block_rate, 4),
        "encoded_attacks": encoded_total,
        "encoded_blocked": encoded_caught,
        "encoded_block_rate": round(encoded_block_rate, 4),
        "normal_total": normal_total,
        "normal_blocked": normal_blocked,
        "normal_pass_rate": round(normal_pass_rate, 4),
        "score_breakdown": {
            "ip_blocking": round(ip_score, 4),
            "encoded_payload_defense": round(encoded_score, 4),
            "jailbreak_defense": round(defense_score, 4),
        },
    }

   return round(min(0.999, max(0.001, total_score)), 4), details



# ---------------------------------------------------------------------------
# Unified grader
# ---------------------------------------------------------------------------

TASK_GRADERS = {
    "cost_hemorrhage": grade_cost_hemorrhage,
    "pii_leak": grade_pii_leak,
    "jailbreak_cascade": grade_jailbreak_cascade,
}

TASK_NAMES = list(TASK_GRADERS.keys())


def _clamp_exclusive(score: float) -> float:
    """Clamp score to the open interval (0, 1) — strictly between 0 and 1.

    The hackathon grading platform rejects scores of exactly 0.0 or 1.0.
    """
    return round(min(0.9999, max(0.0001, score)), 4)


def grade_task(task_name: str, config: GatewayConfig) -> Tuple[float, Dict]:
    """Run the grader for a specific task."""
    grader = TASK_GRADERS.get(task_name)
    if grader is None:
        raise ValueError(f"Unknown task: {task_name}. Must be one of {TASK_NAMES}")
    score, details = grader(config)
    return _clamp_exclusive(score), details
