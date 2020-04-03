from src.Models.BaseModels import ModelBase


class SICRD(ModelBase):
    def __init__(self, a, b, gamma1, gamma2, mu):
        # def __init__(self, a, b, mu):
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.mu = mu
        self.a = a
        self.b = b

    def equation(self, S, I, C, M, RD, RI, t):
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

        gamma2 = (1 - 3 * self.a) / 7
        gamma1 = (1 - 8 * self.mu) / 20
        gamma2 = self.gamma2
        gamma1 = self.gamma1
        dS = -self.b * S * I
        dI = self.b * S * I - (self.a + gamma2) * I
        dC = self.a * I - self.mu * C - gamma1 * C
        dM = self.mu * C
        dRD = gamma1 * C
        dRI = gamma2 * I

        return dS, dI, dC, dM, dRD, dRI
