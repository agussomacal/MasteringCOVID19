from src.Models.BaseModels import ModelBase


class SEIRM(ModelBase):
    def __init__(self, a, b, c, d, m):
        self.a = a
        self.b = b
        self.c = c
        self.m = m
        self.d = d

    def equation(self, S, E, I, R, M, t):
        dS = -self.a * S * E - self.b * S * I

        dE = self.a * S * E - self.c * E
        dI = self.b * S * I - self.m * I - self.d * I

        dR = self.c * E + self.d * I
        dM = self.m * I
        return dS, dE, dI, dR, dM
