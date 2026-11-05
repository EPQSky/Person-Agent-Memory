from __future__ import annotations

from pathlib import Path

from personal_agent_memory.mem2_kernel import Mem2MemoryKernel


class _CountingKernel(Mem2MemoryKernel):
    computes = 0

    def _compute_graph_status(self, library_id: str | None = None) -> dict[str, object]:
        type(self).computes += 1
        return super()._compute_graph_status(library_id)


def test_aggregate_graph_status_is_cached_while_library_probe_stays_exact(
    tmp_path: Path,
) -> None:
    """Polling /status must not pay for the full scope scan on every request."""
    _CountingKernel.computes = 0
    kernel = _CountingKernel(tmp_path / "state")

    first = kernel.graph_status()
    second = kernel.graph_status()

    assert _CountingKernel.computes == 1
    assert first["status"] == second["status"]
    assert first["projected_documents"] == second["projected_documents"]

    kernel.graph_status("user")
    assert _CountingKernel.computes == 2, "explicit library probes must stay exact"


def test_graph_status_cache_can_be_disabled_and_invalidated(tmp_path: Path) -> None:
    _CountingKernel.computes = 0
    uncached = _CountingKernel(tmp_path / "uncached", graph_status_cache_seconds=0)
    uncached.graph_status()
    uncached.graph_status()
    assert _CountingKernel.computes == 2

    _CountingKernel.computes = 0
    cached = _CountingKernel(tmp_path / "cached")
    cached.graph_status()
    assert _CountingKernel.computes == 1
    cached._invalidate_graph_status_cache()
    cached.graph_status()
    assert _CountingKernel.computes == 2
