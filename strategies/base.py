from __future__ import annotations

class Strategy:
    env: object

    def __init__(self, env):
        self.env = env

    def step(self):
        raise NotImplemented('Abstract class')

    @classmethod
    def strategies(cls):
        result = {}
        queue = list(cls.__subclasses__())
        while queue:
            subcls = queue.pop(0)
            result[subcls.__name__] = subcls
            queue.extend(subcls.__subclasses__())
        return result
