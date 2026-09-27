from connections.environment.actions import UndoAction

from .search import SATCoPCon


class SATResetCoP(SATCoPCon):
    """Restart at dead ends, retaining the shadow and its ground instances."""

    def _backtrack(self, state):
        app_id = state.tableau.root.applied_rule_application_id
        if app_id is None:
            return ()
        self._restart()
        return (UndoAction(app_id),)

    def _next_iteration(self):
        self._restart()
        return True

    def _restart(self):
        if self._shadow.consume_new_tableau_clause():
            self.depth_limit -= 1
        self._start_next_depth()
        self._clear_iteration()
