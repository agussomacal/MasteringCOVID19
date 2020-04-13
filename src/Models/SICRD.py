from src.Models.BaseModels import ModelBase


class SICRD(ModelBase):
    # def __init__(self, a, b, gamma1, gamma2, mu, t0, bfactor):
    def __init__(self, b, gamma1, mu, t0, bfactor):

        # def __init__(self, a, b, mu):
        self.gamma1 = gamma1
        self.mu = mu
        self.b = b
        self.t0 = t0
        self.bfactor = bfactor

    # def equation(self, S, I, C, M, RD, RI, t):
    def equation(self, S, C, M, RD, t):
        # S susceptibles
        # I Infectadxs que no estan en cuarentena
        # C Infectadxs Detectadxs
        # M Muertxs
        # RD Recuperadxs detectadxs
        # RI Recuperadxs bo detectadxs

        # if t < 28:
        #     R = R0
        # if t >= 28:
        #     R = k * R0
        #     if t >= 40:
        #         R = R0
        # b = R / Tinc

        # gamma2 = (1 - 3 * self.a) / 7
        # gamma1 = (1 - 8 * self.mu) / 20

        # b = self.b * (1 if t < self.t0 else self.bfactor)
        # dS = -b * S * I
        # dI = b * S * I - (self.a + self.gamma2) * I
        # dC = self.a * I - self.mu * C - self.gamma1 * C
        # dM = self.mu * C
        # dRD = self.gamma1 * C
        # dRI = self.gamma2 * I
        #
        # return dS, dI, dC, dM, dRD, dRI

        b = self.b * (1 if t < self.t0 else self.bfactor)
        dS = -b * S * C
        dC = self.b * C - self.mu * C - self.gamma1 * C
        dM = self.mu * C
        dRD = self.gamma1 * C

        return dS, dC, dM, dRD

