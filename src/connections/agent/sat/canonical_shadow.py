"""SATCoP's incremental random walk, with the existing CDCL/core backend."""

import random

from .shadow import _ShadowSAT

class CanonicalShadow(_ShadowSAT):
    def __init__(self, *, debug_unsat_core = False, seed = 0):
        super().__init__(debug_unsat_core = debug_unsat_core)
        self.rng = random.Random(0 if seed is None else seed)
        self.walk_clauses = []
        self.watches = {}
        self.unsatisfied = []
        self.forced = set()

    def atom_id(self, key):
        identifier = super().atom_id(key)
        self.model.setdefault(identifier, False)
        return identifier

    def add_clause(self, clause, *, clause_idx, from_tableau):
        duplicate = clause in self.clause_contents
        super().add_clause(clause, clause_idx = clause_idx, from_tableau = from_tableau)
        if duplicate or not self.satisfiable:
            return

        index = len(self.walk_clauses)
        self.walk_clauses.append(clause)
        if len(clause) == 1:
            literal = clause[0]
            self.forced.add(abs(literal))
            if self.model[abs(literal)] != (literal > 0):
                self.flip(abs(literal))

        if not self.satisfy(index):
            self.unsatisfied.append(index)

        self.solve()

    def satisfy(self, index):
        for literal in self.walk_clauses[index]:
            if self.model[abs(literal)] == (literal > 0):
                self.watches.setdefault(abs(literal), []).append(index)
                return True

        return False

    def flip(self, variable):
        self.model[variable] = not self.model[variable]
        for index in reversed(self.watches.pop(variable, [])):
            if not self.satisfy(index):
                self.unsatisfied.append(index)

    def choose_unsatisfied(self):
        while self.unsatisfied:
            position = self.rng.randrange(len(self.unsatisfied))
            index = self.unsatisfied[position]
            self.unsatisfied[position] = self.unsatisfied[-1]
            self.unsatisfied.pop()
            if not self.satisfy(index):
                return index

        return None

    def solve(self):
        if not self.dirty or not self.satisfiable:
            return self.satisfiable

        for _ in range(10000):
            index = self.choose_unsatisfied()
            if index is None:
                self.dirty = False
                return True

            possible = [abs(lit) for lit in self.walk_clauses[index] if abs(lit) not in self.forced]
            if not possible:
                self.unsatisfied.append(index)
                break

            self.flip(self.rng.choice(possible))
            self.satisfy(index)

        old_model = dict(self.model)
        if not super().solve():
            return False

        assignment = self.model
        self.model = old_model
        for variable in self.atom_ids.values():
            if variable not in self.forced:
                if self.solver.fixed(variable):
                    self.forced.add(variable)
                if self.model[variable] != assignment.get(variable, False):
                    self.flip(variable)

        return True
