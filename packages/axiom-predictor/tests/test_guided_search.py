from dataclasses import replace
import json

import pytest

from connections.agent import AgentStatus
from connections.clausification import matrix_from_file
from connections.environment.state import State
from connections.environment.tableau import Tableau
from connections.interaction.rollout import rollout

from axiom_prediction.guided import AxiomGuidedSATCoP, AxiomGuidedSATResetCoP, matrix_digest
from axiom_prediction.run import RunConfig, run_problem
from axiom_prediction.training import AxiomTrainingConfig, train_axiom_predictor
from axiom_prediction.wandb_tracking import WandbConfig
from axiom_prediction.cli import main

@pytest.mark.parametrize("agent_class", [AxiomGuidedSATCoP, AxiomGuidedSATResetCoP])
def test_restricted_shadow_and_episode_reuse(tmp_path, agent_class):
    path = tmp_path / "problem.p"
    path.write_text("cnf(a,axiom,p).\ncnf(b,axiom,~p).\ncnf(c,axiom,q).\n")
    matrix = matrix_from_file(path)
    agent = agent_class(
        clause_weights = {0: 0.9, 1: 0.8, 2: 0.1},
        mode = "strict",
        allowed_clause_ids = (0, 2),
        matrix_digest = matrix_digest(matrix),
        debug_sat_core = True,
    )

    for _ in range(2):
        result = rollout(State(matrix, Tableau()), agent, step_limit = 30)
        assert result.status is not AgentStatus.CLOSED
        assert {i for i, _ in agent._shadow.clauses} <= {0, 2}
        assert agent.diagnostics() == {}

    agent = agent_class(clause_weights = {}, mode = "strict", allowed_clause_ids = ())
    assert rollout(
        State(matrix, Tableau()),
        agent,
        step_limit = 10,
    ).status is AgentStatus.GAVE_UP

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
        ),
        wandb_config = WandbConfig(enabled = False),
    )

    config = RunConfig(
        policy = policy,
        checkpoint = str(tmp_path),
        timeout_seconds = 10,
        device = "cpu",
    )

    for mode in ("strict", "weighted"):
        result = run_problem(
            str(tiny_problem_path),
            tptp_root = None,
            config = replace(config, mode = mode),
        )

        assert result["proved"], result

    for model_args in ([], ["--model", str(tmp_path), "--device", "cpu"]):
        capsys.readouterr()
        assert main(
            [
            "run", str(tiny_problem_path), "--policy", policy,
            "--num-workers", "1", "--no-wandb", *model_args,
        ]
        ) == 0

        result = json.loads(capsys.readouterr().out.strip())
        assert result["proved"]
        assert result["mode"] == ("weighted" if model_args else "base")
