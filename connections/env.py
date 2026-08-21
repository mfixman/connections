import dataclasses
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Optional

from connections.calculi.classical import ConnectionState
from connections.calculi.classicalcut import ClassicalCut
from connections.calculi.leancop2 import LeanCoP2State
from connections.calculi.classicalsat import SATConnectionState, SATSolver
from connections.utils.classical_problem_parsing import file2cnf as CNF


class UncaseEnum(StrEnum):
    @classmethod
    def _missing_(cls, value):
        try:
            return next(x for x in cls if x.name.lower() == value.lower())
        except StopIteration as exc:
            raise LookupError(f'"{value}" not a valid {cls.__name__}') from exc

    def __str__(self):
        return self.name

class Logic(UncaseEnum):
    ClassicalSAT = auto()
    Classical = auto()
    ClassicalCut = auto()
    LeanCoP2 = auto()

    def state_class(self):
        class_map = {
            Logic.ClassicalSAT: SATConnectionState,
            Logic.Classical: ConnectionState,
            Logic.ClassicalCut: ClassicalCut,
            Logic.LeanCoP2: LeanCoP2State,
        }

        return class_map[self]

    def parser(self):
        return CNF

    def overrides(self):
        match self:
            case Logic.ClassicalSAT | Logic.ClassicalCut:
                return dict(
                    positive_start_clauses = False,
                    iterative_deepening = True,
                )
            case Logic.LeanCoP2:
                return dict(
                    positive_start_clauses = True,
                    iterative_deepening = True,
                    connection_constraints = False,
                )

        return {}

@dataclass
class Settings:
    positive_start_clauses: bool = True
    iterative_deepening: bool = False
    iterative_deepening_initial_depth: int = 1
    restricted_backtracking: bool = False
    backtrack_after: int = 2
    random_seed: int = 0
    rust_start_clauses: bool = False
    logic: Logic = Logic.Classical
    sat_solver: SATSolver = SATSolver.CaDiCaL
    debug: bool = False
    connection_constraints: bool = True
    time_limit_seconds: Optional[float] = None
    step_limit: Optional[int] = None
    leancop_cut: bool = False
    leancop_scut: bool = False
    leancop_comp_limit: Optional[int] = None
    leancop_conjecture_start: bool = False

    def __post_init__(self):
        self.logic = Logic(self.logic)
        self.sat_solver = SATSolver(self.sat_solver)

class ConnectionEnv:
    def __init__(self, path: str, settings: Optional[Settings] = None):
        if settings is None:
            settings = Settings()

        self.logic = settings.logic
        settings = dataclasses.replace(settings, **self.logic.overrides())

        self.path = path
        self.settings = settings
        self._parse_matrix(self.path)
        self._init_state()

    def _parse_matrix(self, path: str):
        self.matrix = self.logic.parser()(path)

    def _init_state(self):
        self.state = self.logic.state_class()(self.matrix, self.settings)

    def legal_actions(self):
        if self.state.goal is None:
            return [None]

        actions = list(self.state.goal.actions.values())
        if hasattr(self.state, 'control_actions'):
            actions.extend(self.state.control_actions())
        return actions

    @property
    def action_space(self):
        return self.legal_actions()

    def step(self, action):
        if not self.state.is_terminal:
            self.state.update_goal(action)

        status = None if not self.state.info else self.state.info
        return self.state, int(self.state.is_terminal), self.state.is_terminal, status

    def reset(self):
        self.matrix.reset()
        if hasattr(self.state, 'init'):
            self.state.init()
        else:
            self.state.reset()
        return self.state
