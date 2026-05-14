from __future__ import annotations

from importlib import import_module

PUBLIC_STRATEGY_NAMES = (
    'PyCoP',
    'LeanCoP',
    'SATCoP_0',
    'SATCoP',
    'SATResetCoP',
)

HIDDEN_STRATEGY_MODULES = (
    'strategies.hidden.classical',
    'strategies.hidden.sat_reset',
    'strategies.hidden.sat_solve_guided',
    'strategies.hidden.sat_solve_learn',
    'strategies.hidden.sat_solve_propagation',
    'strategies.hidden.sat_solve_rust',
    'strategies.hidden.sat_solve_start',
    'strategies.hidden.sat_solve_ties',
)

class Strategy:
    env: object

    def __init__(self, env):
        self.env = env

    def step(self):
        raise NotImplementedError('Abstract class')

    @classmethod
    def strategies(cls, include_hidden: bool = False):
        if include_hidden:
            for module_name in HIDDEN_STRATEGY_MODULES:
                import_module(module_name)
        result = {}
        queue = list(cls.__subclasses__())
        while queue:
            subcls = queue.pop(0)
            result[subcls.__name__] = subcls
            queue.extend(subcls.__subclasses__())
        if include_hidden:
            return result
        return {
            name: result[name]
            for name in PUBLIC_STRATEGY_NAMES
            if name in result
        }
