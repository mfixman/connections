from typing import Any, Dict, List, Optional, Tuple

from connections.env import Settings
from connections.utils.unification import Substitution
from connections.utils.primitives import Literal


class Tableau:
    """A node in the proof tableau.

    Represents a node in the proof tree, containing a literal and its relationships
    to parent and child nodes.

    Attributes:
        literal_idx: The index of the literal at this node.
        copy_num: The number of the copy of the literal at this node.
        path: The path from the root to this node.
        depth: The depth of this node in the tableau.
        children: The child nodes of this node.
    """

    def __init__(
        self, 
        literal_idx: Optional[tuple[int, int]] = None, 
        clause_copy_num: int = 0, 
        path: Optional[Dict[Tuple[bool, str], List["Tableau"]]] = None,
        parent: Optional["Tableau"] = None
    ) -> None:
        """Initialize a tableau node.

        Args:
            literal_idx: The index of the literal at this node.
            copy_num: The number of the copy of the literal at this node.
            path: The polarity, predicate symbol, and node reference for each node in the path from the root to this node.
        """
        self.literal_idx = literal_idx
        self.copy_num = clause_copy_num
        self.path = path
        self.parent = parent
        self.children: List["Tableau"] = []
        self.closed_branches: int = 0
        
        if path is not None and len(path) > 0:
            self.depth = path[-1].depth + 1
        else:
            self.depth = -1

    def __str__(self) -> str:
        """Return a string representation of the tableau node and its children.

        Returns:
            A formatted string showing the tableau structure.
        """
        angle = "└── " if self.depth >= 0 else ""
        ret = "    " * (self.depth) + angle + str(self.literal) + "\n"
        for child in self.children:
            ret += str(child)
        return ret
    
    def propogate_closed(self) -> None:
        """Propogate the closed branch status to the parent node."""
        self.closed_branches += 1
        if len(self.closed_branches) >= len(self.children):
            self.parent.propogate_closed()


class ConnectionState:
    """State representation for classical connection calculus.

    Maintains the current state of the proof search, including the tableau,
    substitutions, and available actions.

    Attributes:
        matrix: The CNF matrix being proven.
        settings: Configuration settings for the proof search.
        max_depth: Maximum depth for iterative deepening.
        tableau: The current proof tableau.
        goal: The current goal node.
        substitution: Current variable substitutions.
        info: Information about the proof status.
        is_terminal: Whether the proof search is complete.
        proof_sequence: Sequence of actions taken in the proof.
    """

    def __init__(self, matrix: Any, settings: Settings) -> None:
        """Initialize the connection state.

        Args:
            matrix: The CNF matrix to prove.
            settings: Configuration settings for the proof search.
        """
        self.matrix = matrix
        self.settings = settings
        self.reset()

    def __str__(self) -> str:
        """Return a string representation of the current state.

        Returns:
            A formatted string showing the tableau, substitutions, and available actions.
        """
        substitution = "\n".join(
            f"{k} → {v}" for k, v in self.substitution.to_dict().items()
        )
        actions = None
        if self.goal is not None:
            actions = "\n".join(
                f"{i}. {str(action)}"
                for i, action in enumerate(self.goal.actions.values())
            )
        return (
            f"=========================\n"
            f"Tableau:\n{self.tableau}\n\n"
            f"Substitution:\n{substitution}\n\n"
            f"Available Actions:\n{actions}"
            f"\n\nMax Depth: {self.max_depth}"
            f"\n========================="
        )

    def reset(self, depth: Optional[int] = None) -> None:
        """Reset the state to its initial configuration.

        Args:
            depth: Optional maximum depth for iterative deepening.
        """
        # Tableau fields
        self.max_depth = depth
        if depth is None:
            self.max_depth = self.settings.iterative_deepening_initial_depth
        self.tableau = Tableau()
        self.substitution = Substitution()
        
        # Legal start actions
        self.legal_actions: Dict[Tableau, List[ConnectionAction]] = {}
        self.legal_actions[self.tableau] = self._legal_actions(self.tableau)
        if not self.legal_actions[self.tableau]:
            self.info = "Non-Theorem: no positive start clauses"
            self.is_terminal = True
            return

        # Proof fields
        self.info = None
        self.is_terminal = False

    def _starts(self, node: Tableau) -> List["ConnectionAction"]:
        """Get the list of possible start actions.

        Returns:
            List of possible start actions.
        """
        start_clause_candidates = self.matrix.clauses
        if self.settings.positive_start_clauses:
            start_clause_candidates = self.matrix.positive_clauses
            
        return [ConnectionAction(action_type="start", 
                                 principle_node=node, 
                                 connection_idx=i) for i in start_clause_candidates]

    def _undo(self, node: Tableau) -> List["ConnectionAction"]:
        """Get the list of possible backtrack actions.

        Returns:
            List containing the backtrack action.
        """
        return [ConnectionAction(action_type="undo", principle_node=node)]

    def _extensions(self, node: Tableau) -> List["ConnectionAction"]:
        """Get the list of possible extension actions.

        Returns:
            List of possible extension actions.
        """
        extensions = []
        for clause_idx, lit_idx in self.matrix.complements(node.literal_idx):
            
            lit_copy_num = self.matrix.copy_counter[clause_idx] + 1
            unifies, updates = self.substitution.can_unify(
                (node.literal_idx, node.copy_num), (lit_idx, lit_copy_num)
            )
            
            if unifies:
                extensions.append(
                    ConnectionAction(
                        action_type="extension",
                        principle_node=node,
                        lit_idx=(lit_idx, lit_copy_num)
                    )
                )
        return extensions

    def _reductions(self, node: Tableau) -> List["ConnectionAction"]:
        """Get the list of possible reduction actions.

        Returns:
            List of possible reduction actions.
        """
        reductions = []
        for path_node in node.path[(not node.literal.neg, node.literal.symbol)]:
            unifies, updates = self.substitution.can_unify((node.literal_idx, node.copy_num), 
                                                           (path_node.literal_idx, path_node.copy_num))
            if unifies:
                reductions.append(
                    ConnectionAction(
                        action_type="reduction",
                        principle_node=node,
                        connection_idx=(path_node.literal_idx, path_node.copy_num)
                    )
                )
        return reductions

    def _regularizable(self, node: Tableau) -> bool:
        """Check if a clause is regularizable.

        Args:
            clause: The clause to check.

        Returns:
            True if the clause is regularizable, False otherwise.
        """
        for clause_node in node.parent.children:
            clause_lit = self.matrix[clause_node.literal_idx]
            for path_node in node.path[(clause_lit.neg, clause_lit.symbol)]:
                if self.substitution.equal((path_node.literal_idx, path_node.copy_num),
                                           (clause_node.literal_idx, clause_node.copy_num)):
                    return True
        return False

    def _legal_actions(self, node: Tableau) -> Dict[str, "ConnectionAction"]:
        """Get the dictionary of legal actions in the current state.

        Returns:
            Dictionary mapping action IDs to their corresponding actions.
        """
        # Root node
        if node.path == None:
            return self._starts(node)
        
        # Non-root node
        if self._regularizable(node):
            return []
        
        if self.settings.iterative_deepening and (node.depth >= self.max_depth):
            return self._reductions(node) + self._backtracks(node)
        
        return self._reductions(node) + self._extensions(node) + self._backtracks(node)
    
    def _gather_legal_actions(self, node: Tableau) -> None:
        if node.children or node.closed_branches >= len(node.children):
            self.legal_actions[node] = self._undo(node)
            for child in node.children:
                self._gather_legal_actions(child)
                
        else:
            self.legal_actions[node] = self._legal_actions(node)


    def update_goals(self, action: "ConnectionAction") -> None:
        """Update the state based on the selected action.

        Args:
            action: The action to apply.
        """
        self.substitution.update(action.sub_updates)
        
        if action.type == "undo":
            action.principle_node.children = []
            action.principle_node.closed_branches = 0
            self.gather_legal_actions(self.tableau)
            return

        matrix_idx, copy_num = action.connection_idx
        if action.type in ["st", "ex"]:
            (connection_idx, clause_idx) = matrix_idx
            self.matrix.copy_counter[clause_idx] += 1
            
            if action.action_type == "st":
                path = {} 
            else:
                parent_lit = self.matrix[action.principle_node.literal_idx]
                path = action.principle_node.path + {(parent_lit.neg, parent_lit.symbol): [action.principle_node]}
            
            children = [Tableau(lit_idx,copy_num, path, action.principal_node) for lit_idx in self.matrix[clause_idx]]
            children[connection_idx].propogate_closed()
            
        if action.type == "re":
            children = [Tableau(matrix_idx,copy_num, action.principle_node.path, action.principal_node)]
            children[0].propogate_closed()
            
        action.principle_node.children = children

        self.theorem_or_next()

    def theorem_or_next(self) -> None:
        """Check if a theorem has been proven or find the next goal.

        Updates the state's terminal status and goal node accordingly.
        """
        if self.tableau.closed_branches >= len(self.tableau.children):
            self.info = "Theorem"
            self.is_terminal = True
            return
        
        for goal in self.goals:
            goal.actions = self._legal_actions(goal)


class ConnectionAction:
    """Represents an action in the connection calculus proof search.

    Attributes:
        type: The type of action (start, extension, reduction, or backtrack).
        principle_node: The node this action is applied to.
        connection_idx: The matrix index of the connecting literal. ((lit_idx, clause_idx), copy_idx)
    """
    def __init__(
        self,
        action_type: str,
        principle_node: Tableau = None,
        connection_idx: Optional[Tuple[Tuple[int, int], int]] = None,
    ) -> None:
        self.action_type = action_type
        self.principle_node = principle_node
        self.connection_idx = connection_idx

    def __repr__(self):
        return f"{self.principle_node} -> {self.action_type} -> {self.connection_idx}"
