from connections.utils.primitives import *

class Substitution:
    """
    Union-Find w. incremental updates and backtracking
    """
    def __init__(self):
        self.parent = {}
        self.trail = []

    def find(self, item, add=True):
        if not isinstance(item, Variable):
            return item
        if item not in self.parent:
            if not add:
                return item
            self.trail[-1].append(item)
            self.parent[item] = item
            #self.rank[item] = 0
            return item
        if self.parent[item] != item:
            # Path compression
            old = self.parent[item]
            self.parent[item] = self.find(self.parent[item], add)
            if old != self.parent[item]:
                self.trail[-1].append((item, old, self.parent[item]))
        return self.parent[item]

    def union(self, s, t):
        self.trail.append([])
        equations = [(s, t)]

        while equations:
            s, t = equations.pop()
            s = self.find(s)
            t = self.find(t)

            if s == t:
                continue
            
            if isinstance(s, Variable):
                if self.occurs_check(s, t):
                    return False
                self.trail[-1].append((s, self.parent[s], t))
                self.parent[s] = t
            elif isinstance(t, Variable):
                if self.occurs_check(t, s):
                    return False
                self.trail[-1].append((t, self.parent[t], s))
                self.parent[t] = s
            else:
                if s.symbol != t.symbol or len(s.args) != len(t.args):
                    return False
                for arg1, arg2 in zip(s.args, t.args):
                    equations.append((arg1, arg2))  # Add each pair of arguments to the set
        return True
    
    def occurs_check(self, var, term):
        term_root = self.find(term, add=False)
        if var == term_root:
            return True

        if isinstance(term_root, Expression):
            return any(self.occurs_check(var, arg) for arg in term_root.args)

        return False
    
    def backtrack(self):
        updates = self.trail.pop()
        for action in reversed(updates):
            if isinstance(action, Variable):
                var = action
                del self.parent[var]
                continue
            var, old_state, _ = action
            self.parent[var] = old_state
    
    def update(self, update):
        self.trail.append(update)
        for action in update:
            if isinstance(action, Variable):
                var = action
                self.parent[var] = var
                continue
            var, _, new_state = action
            self.parent[var] = new_state

    def can_unify(self, s, t):
        unify, updates = self.unify(s, t)
        self.backtrack()
        return unify, updates
    
    def unify(self, s, t):
        unify = self.union(s, t)
        updates = self.trail[-1]
        return unify, updates
    
    def equal(self, s, t):
        s = self.find(s, add=False)
        t = self.find(t, add=False)
        if s == t:
            return True
        if isinstance(s, Expression) and isinstance(t, Expression):
            return s.symbol == t.symbol and len(s.args) == len(t.args) and all(self.equal(arg1, arg2) for arg1, arg2 in zip(s.args, t.args))
        return False
    
    def __call__(self, term):
        term_root = self.find(term, add=False)
        if isinstance(term_root, Variable):
            return term_root
        return type(term_root)(term_root.symbol,
                          [self(arg) for arg in term_root.args],
                          term_root.prefix)
    
    def to_dict(self):
        substitutions = {}
        for var in self.parent:
            if isinstance(var, Variable) and var in self.parent:
                term = self.find(var, add=False)
                if term != var:
                    substitutions[var] = term
        return substitutions
    
    def __repr__(self):
        return repr(self.to_dict())
