from connections.calculi.classical import ConnectionState, Extension, Reduction, Start
from connections.env import ConnectionEnv, Settings
from connections.utils.primitives import Literal, Matrix


class TestClassicalState:
    def test_initial_state_uses_classical_runtime(self):
        env = ConnectionEnv('tests/cnf_problems/SYN081+1.cnf')

        assert isinstance(env.state, ConnectionState)
        assert isinstance(env.state.matrix, Matrix)

    def test_start_step_opens_a_goal_literal(self):
        env = ConnectionEnv('tests/cnf_problems/SYN081+1.cnf')

        action = env.action_space[0]
        observation, reward, done, info = env.step(action)

        assert isinstance(action, Start)
        assert isinstance(observation.goal.literal, Literal)
        assert reward == 0
        assert done is False
        assert info is None
        assert observation.goal.depth == 0
        assert observation.tableau.literal is None
        assert len(observation.tableau.children) == 2

    def test_classical_goal_exposes_forward_actions_after_start(self):
        env = ConnectionEnv('tests/cnf_problems/SYN081+1.cnf')
        env.step(env.action_space[0])

        actions = env.action_space

        assert any(isinstance(action, Reduction) for action in actions) is False
        assert any(isinstance(action, Extension) for action in actions)
        assert len(actions) >= 3


class TestConnectionEnv:
    def test_iterative_deepening_keeps_current_depth_bound(self):
        env = ConnectionEnv(
            'tests/cnf_problems/SYN081+1.cnf',
            settings=Settings(iterative_deepening=True),
        )

        env.step(env.action_space[0])
        observation, reward, done, info = env.step(env.action_space[0])

        assert reward == 0
        assert done is False
        assert info is None
        assert observation.max_depth == 1
        assert observation.goal.depth == 1

    def test_classical_env_can_take_many_steps_without_api_errors(self):
        env = ConnectionEnv(
            'tests/cnf_problems/GEO081+1.p',
            settings=Settings(iterative_deepening=False),
        )

        for _ in range(200):
            observation, reward, done, info = env.step(env.action_space[0])
            assert reward in {0, 1}
            if done:
                assert info is not None
                break
        else:
            assert observation is not None
