from tabnanny import verbose

import cvxpy
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt

A = np.array([[1, 0],[0, 1]])
B = np.array([[1, 0],[0, 1]])

xinit = np.array([[0], [0]])

uinit = np.array([[0], [0]])

xref = np.array([[2.0], [3.0]])
uref = np.array([[0.0], [0.0]])

(nx, nu) = B.shape

Q = np.eye(nx)

R = np.eye(nu)

P = np.eye(nx)
 
xmin = np.array([[-6], [-6]])  # state constraints
xmax = np.array([[5], [5]])  # state constraints

umax = np.array([[1], [1]])
umin = np.array([[-4],[-4]])

T = 1
N = 50

def linearMPC(T, A, B, N, Q, R, P, xinit, unint, xref, uref, umax=None, umin=None, xmin=None, xmax=None):
    (nx, nu) = B.shape
    x = cvxpy.Variable((nx, N+1))
    u = cvxpy.Variable((nu, N))

    costlist = 0.0
    constrlist = []

    for i in range(N):
        if i > 0:
            costlist += 0.5*cvxpy.quad_form((x[:, i]-xref[:, 0]), Q)

        costlist += 0.5*cvxpy.quad_form((u[:, i]-uref[:,0]), R)
        constrlist += [x[:, i+1] == A@x[:, i] + B@u[:, i]*T ]  

        if xmax is not None:
            constrlist += [x[:, i] <= xmax[:, 0]]

        if xmin is not None:
            constrlist += [x[:, i] >= umin[:, 0]]
        if umax is not None:
            constrlist += [u[:, i] <= umax[:, 0]]

        if umin is not None:
            constrlist += [u[:, i] >= umin[:, 0]]




    costlist += 0.5*cvxpy.quad_form((x[:, i]-xref[:,0]), P)

    constrlist += [x[:, 0] == xinit[:, 0]]
    constrlist += [u[:, 0] == unint[:, 0]]

    prob = cvxpy.Problem(cvxpy.Minimize(costlist), constrlist)
    prob.solve(verbose=True)

    return x.value, u.value

x, u  = linearMPC(T, A, B, N, Q, R, P, xinit, uinit, xref, uref, umax, umin, xmin, xmax)

rx1 = np.array(x[0, :]).flatten()
rx2 = np.array(x[1, :]).flatten()
ru1 = np.array(u[0, :]).flatten()
ru2 = np.array(u[1, :]).flatten()


flg, ax = plt.subplots(1)
plt.plot(rx1, label="x1")
plt.plot(rx2, label="x2")
plt.plot(ru1, label="u")
plt.plot(ru2, label="u")
plt.legend()
plt.grid(True)


plt.show()
