"""
N-GO Client — EnvClient for connecting to the N-GO environment.
"""

try:
    from openenv.core.env_client import EnvClient, StepResult
except ImportError:
    from openenv.core.env_client import EnvClient, StepResult


class NGOEnv(EnvClient):
    """Client for the Neural-Gateway Orchestrator environment."""
    pass
