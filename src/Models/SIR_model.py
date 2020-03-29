from src.Models.BaseModels import ModelBase


class SIR(ModelBase):
    def __init__(self, b, k):
        self.b = b
        self.k = k

    def equation(self, S, I, R, t):
        dS = -self.b*S*I
        dR = self.k*I
        dI = self.b*S*I - self.k*I
        return dS, dI, dR
