import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
import scipy.optimize

def J(x1ETx2) :
    x1,x2 = x1ETx2
    return x1**2 + 1.5*x2**2 - 3*np.sin(2*x1+x2) + 5*np.sin(x1-x2)

def ImprimeIsovaleur(fct,x2d,y2d):
    x2dETy1d = np.array([x2d,y2d])
    nIso = 75  # Define the number of contours等值线 to be plotted as 21
    plt.contour(x2d, y2d, J(x2dETy1d),
                nIso)  # Plot contours based on f1 function values on the grid, generating 21 contours

    plt.title('Isovaleurs')
    plt.xlabel('Valeurs de x')
    plt.ylabel('Valeurs de y')
    plt.grid()
    plt.axis('square')
    plt.show()

def GradJ(x1ETx2) :  # car root ne peut que accepter un seul paramètre à la fct, donc ici on prend une liste x1ETx2 comme paramètre et donnc
    x1, x2 = x1ETx2  # ses valeurs à x1 et x2 dans la fct
    dx1 = 2*x1 - 6*np.cos(2*x1+x2) + 5*np.cos(x1-x2)
    dx2 = 3*x2 - 3*np.cos(2*x1+x2) - 5*np.cos(x1-x2)
    return [dx1,dx2]

def d2J(x1ETx2) :    # calculer la dérivée de 2nd ordre
    x1, x2 = x1ETx2
    d2J_dxdx = 2 + 12*np.sin(2*x1+x2) - 5*np.sin(x1-x2)
    d2J_dxdy = 6*np.sin(2*x1+x2) + 5*np.sin(x1-x2)
    d2J_dydx = 6*np.sin(2*x1+x2) + 5*np.sin(x1-x2)
    d2J_dydy = 3 + 3*np.sin(2*x1+x2) - 5*np.sin(x1-x2)
    return [[d2J_dxdx, d2J_dxdy], [d2J_dydx, d2J_dydy]]

def PrintSolutions(solutions):
    print("racines trouvées：")
    for sol in solutions:
        print(sol)
    print('')

def NewSelution (sol, solutions, tolerance=1e-6):
    for existing_sol in solutions:
        if np.linalg.norm(sol - existing_sol) < tolerance: # calculer la distance euclidienne, si la distance est inférieur
            return False                                   # à la tolérance, alors on pense que c'est une racine existé
    return True

def CalMatrix_DeterNature(solutions):
    #valProp = []
    for i in range(len(solutions)):
        matrix = np.array(d2J(solutions[i]))
        print('Valeurs propres de la matrice :','\n', matrix[0],'\n',matrix[1])
        valPropTem = np.linalg.eigvals(matrix)
        print('  vp1 = {}'.format(valPropTem[0]))
        print('  vp2 = {}'.format(valPropTem[1]))

        if (valPropTem[0] > 0 and valPropTem[1] > 0):
            print("Val min")
            solutions[i].append(0)
        elif (valPropTem[0] < 0 and valPropTem[1] < 0):
            print("Val max")
            solutions[i].append(1)
        elif (valPropTem[0] * valPropTem[1] < 0):
            print("point selle")
            solutions[i].append(2)
        #valProp.append(valPropTem)
        print('')
    #valProp = np.array(valProp)
    return solutions

def FindMinima(fct,x1d,x2d) :
    minima = []
    for x1 in x1d :
        for x2 in x2d :
            minTmp = scipy.optimize.minimize(fct,[x1,x2])
            if minTmp.success and NewSelution(minTmp.x, minima, tolerance=1e-5):
                minima.append(minTmp.x)
    return np.array(minima)

def VerifyPtMin(solutions,minima,tolerance=1e-5) :
    indicater = False
    for sol in solutions:
        if sol[2] == 0:
            for min in minima :
                if abs(min[0] - sol[0]) < tolerance and abs(min[1] - sol[1]) < tolerance:
                    indicater = True
                    break;
                else :
                    continue
            if indicater :
                print(f"We have already found the point [{sol[0]},{sol[1]}] in the solutions found by scipy.optimize.minimize")
                indicater = False
            else :
                print(f"We haven't found the point [{sol[0]},{sol[1]}]in the solutions found by scipy.optimize.minimize")



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
