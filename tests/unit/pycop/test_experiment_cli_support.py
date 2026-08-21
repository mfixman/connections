from __future__ import annotations

import argparse
import json

import pytest

from provers.pycop.cli_common import (
    add_split_arguments,
    finalize_split_arguments,
    partition_items,
)
from provers.pycop.compare_strategies import (
    ComparisonTask,
    load_partial,
    prepare_partial,
    run_task_matrix,
)


def test_split_defaults_from_slurm(monkeypatch):
    parser = argparse.ArgumentParser()
    add_split_arguments(parser)
    args = parser.parse_args([])
    monkeypatch.setenv("SLURM_ARRAY_TASK_COUNT", "3")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")

    finalize_split_arguments(args, parser)

    assert (args.split, args.part) == (3, 1)
    assert partition_items(tuple(range(8)), split=3, part=1) == (1, 4, 7)


def test_partial_manifest_resume_and_mismatch(tmp_path):
    path = tmp_path / "partial.jsonl"
    manifest = {"record_type": "manifest", "schema": "test", "revision": "a"}

    assert prepare_partial(path, manifest=manifest, overwrite=False) == {}
    with path.open("a", encoding="utf-8") as output:
        output.write(
            json.dumps(
                {
                    "record_type": "result",
                    "path": "/tmp/a.p",
                    "policy": "first",
                }
            )
            + "\n"
        )

    assert prepare_partial(path, manifest=manifest, overwrite=False) == {
        ("/tmp/a.p", "first"): {
            "record_type": "result",
            "path": "/tmp/a.p",
            "policy": "first",
        }
    }
    with pytest.raises(RuntimeError, match="different code/configuration"):
        prepare_partial(
            path,
            manifest=manifest | {"revision": "b"},
            overwrite=False,
        )


def test_partial_ignores_only_truncated_tail(tmp_path):
    path = tmp_path / "partial.jsonl"
    manifest = {"record_type": "manifest", "schema": "test"}
    path.write_bytes((json.dumps(manifest) + "\n{\"record_type\":").encode())

    loaded_manifest, results = load_partial(path)

    assert loaded_manifest == manifest
    assert results == {}


def test_task_supervisor_hard_kills_blocking_policy(tmp_path):
    problem = tmp_path / "tiny.p"
    problem.write_text("fof(c,conjecture,(p => p)).\n", encoding="utf-8")
    task = ComparisonTask(
        path=str(problem),
        policy_spec="tests.fixtures.cli_policies:BlockingPolicy",
        policy_label="blocking",
    )
    emitted = []

    results = run_task_matrix(
        (task,),
        settings=(),
        backtrack="step",
        logic="classical",
        domain="constant",
        source_dirs=(),
        max_steps=10,
        timeout=0.05,
        workers=1,
        on_result=emitted.append,
        grace_seconds=0.05,
    )

    assert results == tuple(emitted)
    assert results[0]["status"] == "Timeout"
    assert results[0]["error_type"] is None
