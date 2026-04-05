"""
N-GO Gateway Engine — Core simulation for the AI Gateway.

Contains the model registry, gateway config, condition evaluation,
middleware pipeline with ReDoS protection, routing engine, and security classifier.
"""

import operator
import re
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

class MockModel(BaseModel):
    """A mock LLM model with cost/capability profile."""
    name: str
    cost_per_1k_tokens: float  # USD
    latency_ms: float  # avg latency in ms
    max_complexity: float  # max complexity it handles well (0.0-1.0)
    description: str


MODEL_REGISTRY: Dict[str, MockModel] = {
    "llama-3-8b-instruct": MockModel(
        name="llama-3-8b-instruct",
        cost_per_1k_tokens=0.01,
        latency_ms=50.0,
        max_complexity=0.4,
        description="Fast, cheap. Good for simple chat and summarization.",
    ),
    "llama-3-70b": MockModel(
        name="llama-3-70b",
        cost_per_1k_tokens=0.10,
        latency_ms=200.0,
        max_complexity=0.8,
        description="Balanced. Good for moderate complexity tasks.",
    ),
    "nemotron-3": MockModel(
        name="nemotron-3",
        cost_per_1k_tokens=0.15,
        latency_ms=350.0,
        max_complexity=1.0,
        description="Highest quality. Best for complex reasoning and analysis.",
    ),
}

DEFAULT_MODEL = "nemotron-3"


# ---------------------------------------------------------------------------
# Condition Schema — Safe AST-free routing conditions
# ---------------------------------------------------------------------------

class ConditionSchema(BaseModel):
    """
    Structured condition for routing rules. No eval() — operators are mapped
    to Python's operator module functions.
    """
    field: Literal[
        "complexity_score", "prompt_length", "department",
        "task_type", "source_ip"
    ]
    operator: Literal["lt", "gt", "le", "ge", "eq", "ne", "in", "not_in"]
    value: Union[float, int, str, List[str]]


# Operator mapping — safe, no eval
OPERATOR_MAP = {
    "lt": operator.lt,
    "gt": operator.gt,
    "le": operator.le,
    "ge": operator.ge,
    "eq": operator.eq,
    "ne": operator.ne,
    "in": lambda a, b: a in b,
    "not_in": lambda a, b: a not in b,
}


def evaluate_condition(condition: ConditionSchema, request: "GatewayRequest") -> bool:
    """Safely evaluate a condition against a request without eval()."""
    field_value = getattr(request, condition.field, None)
    if field_value is None:
        return False
    op_func = OPERATOR_MAP.get(condition.operator)
    if op_func is None:
        return False
    try:
        return op_func(field_value, condition.value)
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Routing Rules
# ---------------------------------------------------------------------------

class RoutingRule(BaseModel):
    """A routing rule that maps conditions to target models."""
    condition: ConditionSchema
    target_model: str
    priority: int = 0  # higher priority = evaluated first
    name: str = ""


# ---------------------------------------------------------------------------
# Middleware Patterns
# ---------------------------------------------------------------------------

class MiddlewarePattern(BaseModel):
    """A regex-based middleware for payload scrubbing."""
    pattern: str  # regex pattern
    replacement: str  # replacement string
    name: str = ""


REGEX_TIMEOUT_MS = 50  # 50ms timeout per regex application


def _run_regex_with_timeout(
    pattern: str, replacement: str, text: str, timeout_s: float = 0.05
) -> Tuple[str, bool, bool]:
    """
    Run re.sub with a timeout to prevent ReDoS attacks.

    Returns: (result_text, was_modified, timed_out)
    """
    result = [text]
    modified = [False]
    error = [None]

    def _apply():
        try:
            compiled = re.compile(pattern)
            new_text = compiled.sub(replacement, text)
            if new_text != text:
                modified[0] = True
            result[0] = new_text
        except re.error as e:
            error[0] = str(e)

    thread = threading.Thread(target=_apply, daemon=True)
    thread.start()
    thread.join(timeout=timeout_s)

    if thread.is_alive():
        # ReDoS detected — thread is stuck in catastrophic backtracking
        return text, False, True

    if error[0]:
        return text, False, False

    return result[0], modified[0], False


# ---------------------------------------------------------------------------
# Gateway Request / Response
# ---------------------------------------------------------------------------

class GatewayRequest(BaseModel):
    """A simulated API request hitting the gateway."""
    request_id: str
    prompt: str
    complexity_score: float  # 0.0-1.0
    prompt_length: int
    department: str
    task_type: str  # "summarize", "translate", "chat", "analyze", "code"
    source_ip: str
    token_count: int
    contains_pii: bool = False
    is_attack: bool = False
    attack_type: Optional[str] = None  # "base64", "rot13", "roleplay", "direct"


class RequestResult(BaseModel):
    """Result of processing a request through the gateway."""
    request_id: str
    routed_model: Optional[str] = None
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    blocked: bool = False
    block_reason: Optional[str] = None
    scrubbed_fields: List[str] = []
    scrubbed_payload: Optional[str] = None
    original_payload: Optional[str] = None
    pii_missed: bool = False
    attack_passed: bool = False


# ---------------------------------------------------------------------------
# Gateway Config — The agent-mutable state
# ---------------------------------------------------------------------------

class GatewayConfig(BaseModel):
    """Complete gateway configuration that the agent can modify."""
    routing_rules: List[RoutingRule] = []
    middleware_patterns: List[MiddlewarePattern] = []
    system_prompts: Dict[str, str] = {}
    blocked_ips: Set[str] = set()
    default_model: str = DEFAULT_MODEL
    daily_budget_usd: float = 100.0


# ---------------------------------------------------------------------------
# Gateway Engine
# ---------------------------------------------------------------------------

class GatewayEngine:
    """
    Core simulation engine. Processes requests through the gateway config.

    Pipeline:
    1. IP blocklist check
    2. Middleware pipeline (PII scrubbing) with ReDoS protection
    3. Routing rules evaluation (ConditionSchema, safe)
    4. Security classification (encoding/jailbreak detection)
    5. Cost calculation
    """

    def __init__(self, config: GatewayConfig):
        self.config = config
        self.total_cost = 0.0
        self.requests_processed = 0
        self.requests_blocked = 0
        self.middleware_timeouts = 0

    def process_request(self, request: GatewayRequest) -> RequestResult:
        """Process a single request through the entire gateway pipeline."""

        # 1. IP blocklist
        if request.source_ip in self.config.blocked_ips:
            self.requests_blocked += 1
            return RequestResult(
                request_id=request.request_id,
                blocked=True,
                block_reason=f"IP {request.source_ip} is blocked",
            )

        # 2. Middleware pipeline (PII scrubbing) with ReDoS protection
        current_payload = request.prompt
        scrubbed_fields = []
        redos_detected = False

        for mw in self.config.middleware_patterns:
            result_text, was_modified, timed_out = _run_regex_with_timeout(
                mw.pattern, mw.replacement, current_payload
            )
            if timed_out:
                self.middleware_timeouts += 1
                redos_detected = True
                # Don't apply this regex, skip it
                continue
            if was_modified:
                scrubbed_fields.append(mw.name)
                current_payload = result_text

        # 3. Route to model via rules
        target_model = self.config.default_model
        sorted_rules = sorted(
            self.config.routing_rules, key=lambda r: r.priority, reverse=True
        )
        for rule in sorted_rules:
            if evaluate_condition(rule.condition, request):
                if rule.target_model in MODEL_REGISTRY:
                    target_model = rule.target_model
                    break

        # 4. Security classification
        blocked_by_security = False
        block_reason = None

        if self._detect_encoded_payload(current_payload):
            blocked_by_security = True
            block_reason = "Encoded payload detected (Base64/ROT13)"

        if not blocked_by_security and self._detect_jailbreak(
            current_payload, target_model
        ):
            blocked_by_security = True
            block_reason = "Prompt injection / jailbreak detected"

        if blocked_by_security:
            self.requests_blocked += 1
            return RequestResult(
                request_id=request.request_id,
                blocked=True,
                block_reason=block_reason,
                scrubbed_fields=scrubbed_fields,
                scrubbed_payload=current_payload,
                original_payload=request.prompt,
                attack_passed=False,
            )

        # 5. Cost calculation
        model = MODEL_REGISTRY[target_model]
        cost = (request.token_count / 1000.0) * model.cost_per_1k_tokens
        self.total_cost += cost
        self.requests_processed += 1

        # Check if PII was present but not scrubbed
        pii_missed = request.contains_pii and len(scrubbed_fields) == 0

        return RequestResult(
            request_id=request.request_id,
            routed_model=target_model,
            cost_usd=cost,
            latency_ms=model.latency_ms,
            blocked=False,
            scrubbed_fields=scrubbed_fields,
            scrubbed_payload=current_payload,
            original_payload=request.prompt,
            pii_missed=pii_missed,
            attack_passed=request.is_attack,
        )

    def _detect_encoded_payload(self, payload: str) -> bool:
        """Check if a system prompt is set that detects encoded payloads."""
        # The agent must deploy system prompts that include encoding detection
        # We check if the config has a pre-flight classifier for this
        for model_name, prompt in self.config.system_prompts.items():
            if "reject_encoded" in prompt.lower() or "base64" in prompt.lower():
                # Check if the payload looks encoded
                if self._looks_encoded(payload):
                    return True
        return False

    def _detect_jailbreak(self, payload: str, target_model: str) -> bool:
        """Check if the system prompt provides jailbreak defense."""
        system_prompt = self.config.system_prompts.get(target_model, "")
        if not system_prompt:
            return False

        # Simple heuristic: if system prompt has defensive instructions,
        # check payload for common jailbreak patterns
        defensive_keywords = [
            "ignore previous", "override", "you are now",
            "pretend you", "act as", "forget your instructions",
            "disregard", "new persona", "jailbreak",
            "reveal.*api.*key", "show.*secret", "print.*key",
        ]

        has_defense = any(
            kw in system_prompt.lower()
            for kw in ["reject", "refuse", "block", "deny", "restrict", "defend"]
        )

        if has_defense:
            payload_lower = payload.lower()
            for pattern in defensive_keywords:
                try:
                    if re.search(pattern, payload_lower):
                        return True
                except re.error:
                    if pattern in payload_lower:
                        return True
        return False

    @staticmethod
    def _looks_encoded(text: str) -> bool:
        """Heuristic: does the text look like it contains Base64 or ROT13?"""
        import base64
        import codecs

        # Check for Base64 patterns (long alphanumeric strings with = padding)
        b64_pattern = r'[A-Za-z0-9+/]{20,}={0,2}'
        if re.search(b64_pattern, text):
            try:
                # Try to decode
                for match in re.finditer(b64_pattern, text):
                    decoded = base64.b64decode(match.group()).decode('utf-8', errors='ignore')
                    if len(decoded) > 10 and decoded.isprintable():
                        return True
            except Exception:
                pass

        # Check for ROT13 markers
        rot13_markers = ["rot13:", "ebg13:", "cipher:"]
        if any(m in text.lower() for m in rot13_markers):
            return True

        return False

    def process_batch(
        self, requests: List[GatewayRequest]
    ) -> List[RequestResult]:
        """Process a batch of requests."""
        return [self.process_request(req) for req in requests]
