import numpy as np


# fonction J :



# Fonction recherche du minimum avec le gradient a pas fixe :
def Gradient_PasFixe(J,d1J,X0,alpha,epsilon,Nmax) :
    # initialisation :
    Xn = X0
    dX = 1
    n = 0
    Xn_vector = [Xn]

    # boucle :
    while ((dX>epsilon) and (n<Nmax)) :
        Xnplus1 = Xn - (alpha*d1J(Xn))
        dX = abs(Xnplus1 - Xn)
        Xn = Xnplus1
        n = n + 1
        Xn_vector.append(Xn)

    # indice de convergence :
    Converged = (dX <= epsilon)

    return Xn_vector, Converged


