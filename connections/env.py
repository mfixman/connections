from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

# Type definitions for better code clarity
LogicType = Literal["classical", "intuitionistic", "D", "T", "S4", "S5"]
DomainType = Literal["constant", "cumulative", "varying"]


@dataclass
class Settings:
    """Configuration settings for the connection environment.

    Attributes:
        positive_start_clauses (bool): Whether to only use positive clauses for starting points.
        iterative_deepening (bool): Whether to use iterative deepening search.
        iterative_deepening_initial_depth (int): Initial depth for iterative deepening.
        restricted_backtracking (bool): Whether to use restricted backtracking.
        backtrack_after (int): Number of attempts before backtracking is restricted.
        logic (LogicType): The logical system to use (classical, intuitionistic, or modal).
        domain (DomainType): The domain type for modal logics.
    """

    positive_start_clauses: bool = True
    iterative_deepening: bool = False
    iterative_deepening_initial_depth: int = 1
    restricted_backtracking: bool = False
    backtrack_after: int = 2
    logic: LogicType = "classical"
    domain: DomainType = "constant"


class ConnectionEnv:
    """A reinforcement learning environment for connection calculi.

    This environment implements the connection calculus proof search as a Markov Decision Process.
    It supports classical, intuitionistic, and various modal logics.

    Attributes:
        path (str): Path to the problem file in CNF format.
        settings (Settings): Configuration settings for the environment.
        matrix: The parsed CNF matrix.
        state: The current state of the proof search.
    """

    def __init__(
        self, path: Union[str, Path], settings: Optional[Settings] = None
    ) -> None:
        """Initialize the connection environment.

        Args:
            path: Path to the problem file in CNF format.
            settings: Optional configuration settings. If None, default settings are used.
        """
        if settings is None:
            settings = Settings()

        self.path = str(path)
        self.settings = settings
        self._parse_matrix(self.path)
        self._init_state()

    def _parse_matrix(self, path: str) -> None:
        """Parse the CNF matrix from the input file.

        Args:
            path: Path to the problem file.
        """
        if self.settings.logic == "classical":
            from connections.utils.cnf_parsing import file2cnf
        else:
            from connections.utils.icnf_parsing import file2cnf
        self.matrix = file2cnf(path)

    def _init_state(self) -> None:
        """Initialize the appropriate state class based on the selected logic."""
        logic_state_map = {
            "classical": "connections.calculi.classical.ConnectionState",
            "intuitionistic": "connections.calculi.intuitionistic.IConnectionState",
            "D": "connections.calculi.modal_d.DConnectionState",
            "T": "connections.calculi.modal_t.TConnectionState",
            "S4": "connections.calculi.modal_s4.S4ConnectionState",
            "S5": "connections.calculi.modal_s5.S5ConnectionState",
        }

        state_class_path = logic_state_map[self.settings.logic]
        module_name, class_name = state_class_path.rsplit(".", 1)
        module = __import__(module_name, fromlist=[class_name])
        state_class = getattr(module, class_name)

        self.state = state_class(self.matrix, self.settings)

    @property
    def action_space(self) -> List[Any]:
        """Get the list of available actions in the current state.

        Returns:
            List of available actions. Returns [None] if no actions are available.
        """
        if self.state.goal is None:
            return [None]
        actions = list(self.state.goal.actions.values())
        return actions

    def step(self, action: Any) -> Tuple[Any, int, bool, Dict[str, str]]:
        """Execute one step in the environment.

        Args:
            action: The action to take.

        Returns:
            A tuple containing:
            - The new state
            - The reward (1 for terminal state, 0 otherwise)
            - Whether the episode is done
            - Additional information about the state
        """
        if self.state.is_terminal:
            return (
                self.state,
                int(self.state.is_terminal),
                self.state.is_terminal,
                {"status": self.state.info},
            )
        else:
            self.state.update_goal(action)
        return (
            self.state,
            int(self.state.is_terminal),
            self.state.is_terminal,
            {"status": self.state.info},
        )

    def reset(self) -> Any:
        """Reset the environment to its initial state.

        Returns:
            The initial state of the environment.
        """
        self.matrix.reset()
        self.state.reset()
        return self.state
