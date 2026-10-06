from dataclasses import replace
import json

import pytest

from axiom_prediction.run import RunConfig, run_problem
from axiom_prediction.training import AxiomTrainingConfig, train_axiom_predictor
from axiom_prediction.wandb_tracking import WandbConfig
from axiom_prediction.cli import main

@pytest.mark.parametrize("policy", ["satcop", "satresetcop"])
def test_trained_checkpoint_guides_search(tmp_path, tiny_problem_path, policy, capsys):
    train_axiom_predictor(
        [str(tiny_problem_path)],
        output_dir = tmp_path,
        config = AxiomTrainingConfig(
            epochs = 1,
            hidden_dim = 8,
            message_rounds = 1,
            device = "cpu",
            sat_policy = policy,
        ),
        wandb_config = WandbConfig(enabled = False),
    )

    config = RunConfig(
        policy = policy,
        checkpoint = str(tmp_path),
        timeout_s = 10,
        device = "cpu",
    )

    for mode in ("strict", "weighted"):
        result = run_problem(
            str(tiny_problem_path),
            tptp_root = None,
            config = replace(config, mode = mode),
        )

        assert result["proved"], result

    other = "satcop" if policy == "satresetcop" else "satresetcop"
    capsys.readouterr()
    result = run_problem(
        str(tiny_problem_path),
        tptp_root = None,
        config = replace(config, policy = other),
    )

    assert result["proved"], result
    assert result["policy"] == other
    assert not result.get("error")
    assert "warning: dataset labels were collected" in capsys.readouterr().err

    for model_args in ([], ["--model", str(tmp_path), "--device", "cpu"]):
        capsys.readouterr()
        assert main(
            [
            "run", str(tiny_problem_path), "--policy", other,
            "--num-workers", "1", "--no-wandb", *model_args,
        ]
        ) == 0

        captured = capsys.readouterr()
        if model_args:
            assert "warning: dataset labels were collected" in captured.err

        records = [json.loads(line) for line in captured.out.splitlines()]
        result = records[0]
        assert len(records) == 1
        assert "event" not in result
        assert result["proved"]
        assert result["policy"] == other
        assert result["mode"] == ("weighted" if model_args else "base")
