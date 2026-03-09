from connections.env import ConnectionEnv

class Strategy:
    env: ConnectionEnv

    def __init__(self, env):
        self.env = env

    def step(self):
        raise NotImplemented('Abstract class')

    @classmethod
    def strategies(cls):
        return {
            'FirstAction': FirstAction,
            'SmartCut': SmartCut,
        }

class FirstAction(Strategy):
    def step(self):
        return self.env.legal_actions()[0]

class SmartCut(Strategy):
    def step(self):
        print(self.env.legal_actions())
        return self.env.legal_actions()[0]
