from typing import Callable, Optional
from connections.utils.primitives import Matrix, Literal, Variable

import logging

class Tableau:
    def __init__(self, literal = None, parent = None):
        self.literal = literal
        self.parent = parent
        self.children = []
        self.proven = False
        # Cut-related state is always present on all nodes.
        self.cut_committed = False
        self.pending_cut = False
        self.depth = parent.depth + 1 if parent is not None else -1
        self.num_attempted = 0
        self.actions = {}

    def __str__(self):
        angle = "└── " if self.depth >= 0 else ""
        ret = ["    " * (self.depth) + angle + str(self.literal)]
        for child in self.children:
            ret.append(str(child))
        return '\n'.join(ret)

    def __repr__(self):
        return str(self)

    def path(self):
        path = []
        current = self.parent
        while current.literal is not None:
            path.append(current.literal)
            current = current.parent
        return path

    def find_best(self, scorer: Callable[[Literal], int]) -> Optional['Tableau']:
        parent = self

        while parent is not None:
            nonproven = []

            for child in parent.children:
                if not child.proven:
                    nonproven.append(child)

            if nonproven:
                # if any(scorer(x.literal) != 0 for x in nonproven):
                    # print('Literal with positive score!')
                    # import ipdb
                    # ipdb.set_trace()

                return max(nonproven, key = lambda x: scorer(x.literal))

            parent.proven = True
            parent = parent.parent

        return None

    # DFS over tree, find next unproven node.
    def find_next(self):
        parent = self

        while parent is not None:
            for child in parent.children:
                if not child.proven:
                    return child

            parent.proven = True
            parent = parent.parent

        return None

    def find_prev(self):
        parent = self.parent
        self_idx = parent.children.index(self)

        if self_idx > 1 or (self_idx == 1 and parent.literal is None):
            prev = parent.children[self_idx - 1]
            while len(prev.children) > 1:
                prev = prev.children[-1]
        else:
            prev = parent
        return prev
