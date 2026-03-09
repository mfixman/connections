from connections.env import ConnectionEnv

class Strategy:
    def step(self):
        raise NotImplemented('Abstract class')

    @classmethod
    def strategies(cls):
        return {
            'FirstAction': FirstAction,
        }

class FirstAction(Strategy):
    env: ConnectionEnv

    def __init__(self, env):
        self.env = env

    def __repr__(self):
        return 'FirstAction'

    def step(self):
        return self.env.legal_actions()[0]
