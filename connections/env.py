import dataclasses
from dataclasses import dataclass
from typing import Optional, Literal

from enum import StrEnum, auto
from Utils import *

from connections.calculi.classicalsat import SATConnectionState
from connections.calculi.classical import ConnectionState
from connections.calculi.intuitionistic import IConnectionState
from connections.calculi.modal_d import DConnectionState
from connections.calculi.modal_t import TConnectionState
from connections.calculi.modal_s4 import S4ConnectionState
from connections.calculi.modal_s5 import S5ConnectionState

from connections.utils.cnf_parsing import file2cnf as CNF
from connections.utils.icnf_parsing import file2cnf as ICNF

class Logic(UncaseEnum):
    ClassicalSAT = auto()
    Classical = auto()
    Intuitionistic = auto()
    Modal = auto()
    D = auto()
    T = auto()
    S4 = auto()
    S5 = auto()

    def state_class(self):
        class_map = {
            Logic.ClassicalSAT: SATConnectionState,
            Logic.Classical: ConnectionState,
            Logic.Intuitionistic: IConnectionState,
            Logic.D: DConnectionState,
            Logic.T: TConnectionState,
            Logic.S4: S4ConnectionState,
            Logic.S5: S5ConnectionState
        }

        return class_map[self]

    def parser(self):
        match self:
            case Logic.Classical | Logic.ClassicalSAT:
                return CNF

            case Logic.Intuitionistic | Logic.D | Logic.T | Logic.S4 | Logic.S5:
                return ICNF

        raise LookupError(f'Unknown parser for logic {self}')

    def overrides(self):
        match self:
            case Logic.ClassicalSAT:
                return dict(
                    positive_start_clauses = False,
                    iterative_deepening = True,
                )

        return {}

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

        settings = dataclasses.replace(settings, **settings.logic.overrides())

        self.path = path
        self.settings = settings
        self._parse_matrix(self.path)
        self._init_state()

    def _parse_matrix(self, path: str):
        self.matrix = self.settings.logic.parser()(path)

    def _init_state(self):
        self.state = self.settings.logic.state_class()(self.matrix, self.settings)

    @property
    def action_space(self):
        if self.state.goal is None:
            return [None]
        actions = list(self.state.goal.actions.values())
        return actions

    def step(self, action):
        if not self.state.is_terminal:
            self.state.update_goal(action)

        status = None if not self.state.info else self.state.info
        return self.state, int(self.state.is_terminal), self.state.is_terminal, status

    def reset(self):
        self.matrix.reset()
        self.state.reset()
        return self.state
