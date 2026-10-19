from __future__ import annotations

import pytest

from adam_impact_study import ray_cluster


def test_explicit_serial_mode_does_not_initialize_ray(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ray_cluster.ray,
        "is_initialized",
        lambda: (_ for _ in ()).throw(AssertionError("Ray should not be inspected")),
    )

    assert ray_cluster.initialize_use_ray(num_cpus=1) is False
    assert ray_cluster.initialize_use_ray(num_cpus=0) is False


def test_connects_to_existing_cluster(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(ray_cluster.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(ray_cluster.ray, "init", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(ray_cluster.ray, "cluster_resources", lambda: {"CPU": 4.0})

    assert ray_cluster.initialize_use_ray(num_cpus=4) is True
    assert calls == [{"address": "auto", "include_dashboard": False}]


def test_falls_back_to_bounded_local_cluster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def _init(**kwargs: object) -> None:
        calls.append(kwargs)
        if kwargs.get("address") == "auto":
            raise ConnectionError("no cluster")

    monkeypatch.setattr(ray_cluster.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(ray_cluster.ray, "init", _init)
    monkeypatch.setattr(ray_cluster.ray, "cluster_resources", lambda: {"CPU": 3.0})

    assert ray_cluster.initialize_use_ray(num_cpus=3, object_store_bytes=1234) is True
    assert calls == [
        {"address": "auto", "include_dashboard": False},
        {
            "num_cpus": 3,
            "object_store_memory": 1234,
            "include_dashboard": False,
        },
    ]
