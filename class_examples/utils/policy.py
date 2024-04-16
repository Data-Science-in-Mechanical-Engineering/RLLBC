import numpy as np

class PDPolicy:
    def __init__(self, env, swingup=True, eps_expl=0.00):
        self.K = np.asarray([[10,  2.0]])
        self.eps_expl = eps_expl
        self.env = env
        self.swingup = swingup

    def get_action(self, x):

        th, thdot = x


        if th < -np.pi/2 and self.swingup:
            u = [np.sign(thdot) - .1*thdot]
        elif th > np.pi/2 and self.swingup:
            u = [np.sign(thdot)  - .1*thdot]
        elif np.random.rand() < self.eps_expl:
            u = self.env.action_space.sample()
        else:
            u = np.dot(-self.K, [th, thdot])

        return np.clip(u,self.env.action_space.low,self.env.action_space.high)[0]
