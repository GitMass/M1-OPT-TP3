import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
import scipy.optimize



## Partie 1 :

def J(x1ETx2) :
    x1,x2 = x1ETx2
    return x1**2 + 1.5*x2**2 - 3*np.sin(2*x1+x2) + 5*np.sin(x1-x2)

def ImprimeIsovaleur(fct,x2d,y2d):
    x2dETy1d = np.array([x2d,y2d])
    nIso = 75  # Define the number of contours等值线 to be plotted as 21
    plt.contour(x2d, y2d, fct(x2dETy1d),
                nIso)  # Plot contours based on f1 function values on the grid, generating 21 contours

    plt.title('Isovaleurs')
    plt.xlabel('Valeurs de x')
    plt.ylabel('Valeurs de y')
    plt.grid()
    plt.axis('square')

def GradJ(x1ETx2) :  # car root ne peut que accepter un seul paramètre à la fct, donc ici on prend une liste x1ETx2 comme paramètre et donnc
    x1, x2 = x1ETx2  # ses valeurs à x1 et x2 dans la fct
    dx1 = 2*x1 - 6*np.cos(2*x1+x2) + 5*np.cos(x1-x2)
    dx2 = 3*x2 - 3*np.cos(2*x1+x2) - 5*np.cos(x1-x2)
    return np.array([dx1,dx2])

def d2J(x1ETx2) :    # calculer la dérivée de 2nd ordre
    x1, x2 = x1ETx2
    d2J_dxdx = 2 + 12*np.sin(2*x1+x2) - 5*np.sin(x1-x2)
    d2J_dxdy = 6*np.sin(2*x1+x2) + 5*np.sin(x1-x2)
    d2J_dydx = 6*np.sin(2*x1+x2) + 5*np.sin(x1-x2)
    d2J_dydy = 3 + 3*np.sin(2*x1+x2) - 5*np.sin(x1-x2)
    return np.array([[d2J_dxdx, d2J_dxdy], [d2J_dydx, d2J_dydy]])

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
            print(f"point : {solutions[i]} is Val min")
            solutions[i].append(0)
        elif (valPropTem[0] < 0 and valPropTem[1] < 0):
            print(f"point : {solutions[i]} Val max")
            solutions[i].append(1)
        elif (valPropTem[0] * valPropTem[1] < 0):
            print(f"point : {solutions[i]} point selle")
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



## Partie 2 :

def Gradient_PasFixe(J,d1J,X0,alpha,epsilon,Nmax) :
    # initialisation :
    Xn = np.array(X0, dtype=float)
    dX = 1
    n = 0
    Xn_vector = [Xn]

    # boucle :
    while ((dX>epsilon) and (n<Nmax)) :
        Xnplus1 = Xn - (alpha * d1J(Xn))
        dX = np.sqrt(((Xnplus1[0] - Xn[0])**2)+((Xnplus1[1] - Xn[1])**2))
        Xn = Xnplus1
        n = n + 1
        Xn_vector.append(Xn)

    # indice de convergence :
    Converged = (dX <= epsilon)
    Xn_vector = np.array(Xn_vector)

    return Xn_vector, Converged



def Gradient_PasOptimal(F,d1F,X0,epsilon,Nmax,printalpha) :

    def F_step(alpha,Xn,grad):
        return F(Xn - (alpha * grad))

    # initialisation :
    Xn = np.array(X0, dtype=float)
    dX = 1
    n = 0
    Xn_vector = [Xn]

    # boucle :
    while ((dX>epsilon) and (n<Nmax)) :
        # Calcul du gradient
        grad = d1F(Xn)

        # calculer le alpha optimal en minimisant F_step
        res = scipy.optimize.minimize_scalar(F_step, args=(Xn, grad), bounds=(0, 1), method='bounded')
        alpha = res.x

        Xnplus1 = Xn - (alpha * grad)
        dX = np.linalg.norm(Xnplus1 - Xn)
        Xn = Xnplus1
        n = n + 1
        Xn_vector.append(Xn)

        if (printalpha==True) :
            print(f"Iteration {n}: alpha={alpha}")

    # indice de convergence :
    Converged = (dX <= epsilon)
    Xn_vector = np.array(Xn_vector)

    return Xn_vector, Converged

def Newton(J,d1J,d2J,X0,epsilon,Nmax) :
    # initialisation :
    Xn = np.array(X0, dtype=float)
    dX = 1
    n = 0
    Xn_vector = [Xn]

    # boucle :
    while ((dX>epsilon) and (n<Nmax)) :
        grad = d1J(Xn)
        hess = d2J(Xn)

        hess_inv = np.linalg.inv(hess)
        X_delta = np.dot(hess_inv, -grad)
        Xnplus1 = Xn + X_delta
        dX = np.linalg.norm(Xnplus1 - Xn)

        Xn = Xnplus1
        n = n + 1
        Xn_vector.append(Xn)

    # indice de convergence :
    Converged = (dX <= epsilon)
    Xn_vector = np.array(Xn_vector)

    return Xn_vector, Converged




## Partie 3 :

def l(xETy):
    x,y = xETy
    return y+3/2*x

def grad_l(xETy):
    x, xy = xETy
    dl_dx1 = 3/2
    dl_dx2 = 1
    return np.array([dl_dx1, dl_dx2])

def StraightEq(a,b,x):
    y = ((x-a[0])/(b[0]-a[0])) * (b[1]-a[1]) + a[1]
    return y

def DrawStraignt(x,y):
    plt.plot(x, y, 'r-', label='Line through A(0,0) and B(2,-3)', linewidth=2)  # red straight
    plt.legend()

def LagrangeEqs(variables):  # ici, on remplace x1 et x2 dans J par x et y, donc c'est x^2 + 1.5y^2 - 3sin(2x+y) + 5sin(x-y)
    x, y, lambd = variables
    eq1 = 2 * x - 6 * np.cos(2 * x + y) + 5 * np.cos(x - y) - 3 / 2 * lambd  # ∂x/∂J - λ*(∂x/∂M)
    eq2 = 3 * y - 3 * np.cos(2 * x + y) - 5 * np.cos(x - y) - lambd  # ∂y/∂J - λ*(∂y/∂M)
    eq3 = 3 / 2 * x + y  # l'équation du droite : M
    return [eq1, eq2, eq3]

def RootLagEq(init_Guess):
    solutions = []
    for i in init_Guess:
        solution = root(LagrangeEqs, i)
        if solution.success:
            solutions.append(solution.x)
    return solutions