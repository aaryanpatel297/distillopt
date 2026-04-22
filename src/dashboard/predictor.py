import random

def predict(z,T,R,P,N,F):

    purity = 0.80 + 0.15*(z) - 0.02*(R-3)**2
    bottoms = 0.12 - 0.05*(z)
    energy = 5000 + 2000*(R) + 10*(F)

    return purity, bottoms, energy