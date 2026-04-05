"""
OptiChain Inventory Environment Client.

Provides a synchronous HTTP client for interacting with the
OptiChain supply chain environment server.
"""

from __future__ import annotations

import requests

from models import SupplyChainAction, SupplyChainObservation, SupplyChainState


class OptiChainEnv:
    """
    HTTP client for the OptiChain supply chain environment.

    Wraps the REST endpoints exposed by server/app.py.

    Example:
        >>> client = OptiChainEnv(base_url="http://localhost:7860")
        >>> obs = client.reset("task_01_easy")
        >>> while not obs.done:
        ...     action = SupplyChainAction(orders=[])
        ...     obs = client.step(action)
        >>> print(client.grader_score())
    """

    def __init__(self, base_url: str = "http://localhost:7860") -> None:
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Core OpenEnv endpoints
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "task_01_easy") -> SupplyChainObservation:
        """Reset the environment and load a task. Returns initial observation."""
        resp = self._session.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id},
            timeout=10,
        )
        resp.raise_for_status()
        return SupplyChainObservation.model_validate(resp.json())

    def step(self, action: SupplyChainAction) -> SupplyChainObservation:
        """Advance the simulation one day. Returns updated observation."""
        resp = self._session.post(
            f"{self.base_url}/step",
            json=action.model_dump(),
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return SupplyChainObservation.model_validate(data["observation"])

    def state(self) -> SupplyChainState:
        """Return current episode metadata without advancing time."""
        resp = self._session.get(f"{self.base_url}/state", timeout=10)
        resp.raise_for_status()
        return SupplyChainState.model_validate(resp.json())

    def grader_score(self) -> float:
        """Return the normalised grader score in [0.0, 1.0]."""
        resp = self._session.get(f"{self.base_url}/grader", timeout=10)
        resp.raise_for_status()
        return float(resp.json()["score"])

    def health(self) -> bool:
        """Return True if the server is healthy."""
        try:
            resp = self._session.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200 and resp.json().get("status") == "healthy"
        except requests.RequestException:
            return False

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "OptiChainEnv":
        return self

    def __exit__(self, *_) -> None:
        self.close()
