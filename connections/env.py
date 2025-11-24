from dataclasses import dataclass
from typing import Optional, Literal

from enum import StrEnum, auto
from Utils import *

class Logic(UncaseEnum):
    Classical = auto()
    Intuitionistic = auto()
    Modal = auto()
    D = auto()
    T = auto()
    S4 = auto()
    S5 = auto()

class Domain(UncaseEnum):
    Constant = auto()
    Cumulative = auto()
    Varying = auto()

@dataclass
class Settings:
    positive_start_clauses: bool = True
    iterative_deepening: bool = False
    iterative_deepening_initial_depth: int = 1
    restricted_backtracking: bool = False
    backtrack_after: int = 2
    logic: Logic = Logic.Classical
    domain: Domain = Domain.Constant

class ConnectionEnv:
    def __init__(self, path: str, settings: Optional[Settings] = None):
        if settings is None:
            settings = Settings()

        self.path = path
        self.settings = settings
        self._parse_matrix(self.path)
        self._init_state()

    def _parse_matrix(self, path: str):
        if self.settings.logic == Logic.Classical:
            from connections.utils.cnf_parsing import file2cnf
        else:
            from connections.utils.icnf_parsing import file2cnf
        self.matrix = file2cnf(path)

    def _init_state(self):
        logic_state_map = {
            Logic.Classical: 'connections.calculi.classical.ConnectionState',
            Logic.Intuitionistic: 'connections.calculi.intuitionistic.IConnectionState',
            Logic.D: 'connections.calculi.modal_d.DConnectionState',
            Logic.T: 'connections.calculi.modal_t.TConnectionState',
            Logic.S4: 'connections.calculi.modal_s4.S4ConnectionState',
            Logic.S5: 'connections.calculi.modal_s5.S5ConnectionState'
        }

        state_class_path = logic_state_map[self.settings.logic]
        module_name, class_name = state_class_path.rsplit('.', 1)
        module = __import__(module_name, fromlist=[class_name])
        state_class = getattr(module, class_name)

        self.state = state_class(self.matrix, self.settings)

    @property
    def action_space(self):
        if self.state.goal is None:
            return [None]
        actions = list(self.state.goal.actions.values())
        return actions

    def step(self, action):
        if self.state.is_terminal:
            return self.state, int(self.state.is_terminal), self.state.is_terminal, {"status": self.state.info}
        else:
            self.state.update_goal(action)
        return self.state, int(self.state.is_terminal), self.state.is_terminal, {"status": self.state.info}

    def reset(self):
        self.matrix.reset()
        self.state.reset()
        return self.state
