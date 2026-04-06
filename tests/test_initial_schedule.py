import json
from pathlib import Path

import strategies

class DummyEnv:
    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

class Alpha(strategies.Strategy):
    def step(self):
        return 'alpha'

class Beta(strategies.Strategy):
    def step(self):
        return 'beta'

class TestInitialSchedule:
    def test_switches_to_next_slot_after_budget_and_resets_env(self, tmp_path: Path, monkeypatch):
        schedule_path = tmp_path / 'schedule.json'
        schedule_path.write_text(
            json.dumps(
                {
                    'slots': [
                        {'strategy': 'Alpha', 'step_budget': 2},
                        {'strategy': 'Beta', 'step_budget': 3},
                    ]
                }
            ),
            encoding = 'utf-8',
        )

        monkeypatch.setattr(strategies.InitialSchedule, 'schedule_path', schedule_path)

        env = DummyEnv()
        strategy = strategies.InitialSchedule(env)

        assert strategy.step() == 'alpha'
        assert strategy.step() == 'alpha'
        assert env.reset_count == 0

        assert strategy.step() == 'beta'
        assert env.reset_count == 1
        assert strategy.step() == 'beta'

    def test_does_not_schedule_itself(self, tmp_path: Path, monkeypatch):
        schedule_path = tmp_path / 'schedule.json'
        schedule_path.write_text(
            json.dumps({'slots': [{'strategy': 'InitialSchedule', 'step_budget': 1}]}),
            encoding = 'utf-8',
        )

        monkeypatch.setattr(strategies.InitialSchedule, 'schedule_path', schedule_path)

        try:
            strategies.InitialSchedule(DummyEnv())
        except ValueError as exc:
            assert 'cannot schedule itself' in str(exc)
        else:
            raise AssertionError('InitialSchedule should reject self-scheduling')
