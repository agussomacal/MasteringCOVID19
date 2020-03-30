from src.Models.BaseModels import ModelBase


class SIRM(ModelBase):
    def __init__(self, b, d, m):
        self.b = b
        self.m = m
        self.d = d

    def equation(self, S, I, R, M, t):
        dS = - self.b * S * I

        dI = self.b * S * I - self.m * I - self.d * I

        dR = self.d * I
        dM = self.m * I
        return dS, dI, dR, dM
