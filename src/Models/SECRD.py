from src.Models.BaseModels import ModelBase


class SECRD(ModelBase):
    def __init__(self, a, b, gamma1, gamma2, mu):
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

        dS = -self.b * S * I
        dI = self.b * S * I - (self.a + self.gamma2) * I
        dC = self.a * I - self.mu * C - self.gamma1 * C
        dM = self.mu * C
        dRD = self.gamma1 * C
        dRI = self.gamma2 * I

        return dS, dI, dC, dM, dRD, dRI
