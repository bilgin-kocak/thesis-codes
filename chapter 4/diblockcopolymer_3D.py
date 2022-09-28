#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 22:23:19 2020

@author: bilginkocak
"""

import random
from dolfin import *
import numpy as np

class InitialConditions(UserExpression):
    def __init__(self, **kwargs):
        random.seed(2 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)
    def eval(self, values, x):
        # values[0] = 0.63 + 0.02*(0.5 - random.random())
        # if (x[0]-0.5)**2 + (x[1]-0.5)**2 <= 0.1:
        #     values[0] = np.sin(10*np.pi*x[0])
        # else:
        #     values[0] = -0.99
        if (x[0]-0.5)**2 + (x[1]-0.5)**2 + (x[2]-0.5)**2 <= 0.125:
            values[2] = np.sin(15*np.pi*x[0])
        else:
            values[2] = -0.2 #+ 0.2*(0.5 - random.random())
        # values[0] = np.sin(10*x[0]*x[1])
        values[1] = 0.0
        # values[2] = np.cos(10*(x[0]-x[1]))*x[0]*x[1]
        # if (x[0]-0.5)**2 + (x[1]-0.5)**2 <= 0.1:
        #     np.cos(10*(x[0]-x[1]))*x[0]*x[1]
        # else:
        #     values[2] = -0.99
        if (x[0]-0.5)**2 + (x[1]-0.5)**2 + (x[2]-0.5)**2 <= 0.125:
            # values[0] = np.cos(10*(x[0]-x[1]))*x[0]*x[1]
            values[0] = +0.6 #+ 0.2*(0.5 - random.random())
        else:
            values[0] = -0.6 #+ 0.2*(0.5 - random.random())
        values[3] = values[2]
        values[1] = values[0]
        # values[0] = 0.0 + 2*(0.5 - random.random())
        # values[1] = 0.0
        # values[2] = 0.0 + 2*(0.5 - random.random())
        # values[3] = 0.0
    def value_shape(self):
        return (4,)

# Form compiler options
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True

# Create mesh and build function space
mesh = UnitCubeMesh.create(80, 80, 80, CellType.Type.tetrahedron)
#mesh = UnitSquareMesh(20,20)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = MixedElement([P1, P1, P1, P1])
ME = FunctionSpace(mesh, TH)

dt = 0.0005  # time step
tend = 0.1
num = int(tend/dt) + 1
time_array = np.linspace(0,tend, num)
epsv = 0.025
epsu = 0.025
tu = 1
tv = 10
beta = -0.8
alpha = 0.3
sigma = 100

# Define trial and test functions
du    = TrialFunction(ME)
q1, q2, q3, q4  = TestFunctions(ME)

# Define functions
uu   = Function(ME)  # current solution
uu0  = Function(ME)  # solution from previous converged step

# Split mixed functions
dc1, dmu1, dc2, dmu2 = split(du)
c1,  mu1, c2, mu2  = split(uu)
c10, mu10, c20, mu20 = split(uu0)

# Create intial conditions and interpolate
u_init = InitialConditions(degree=2)
uu.interpolate(u_init)
uu0.interpolate(u_init)

f1 = (1-c1)*(1+c1)*c1 - alpha*c2 - beta*c2**2
f2 = (1-c2)*(1+c2)*c2 - alpha*c1 - 2*beta*c1*c2
vbar = assemble(uu.split()[2]*dx)
# vbar = 0.0114559
L0 = c1*q1*dx - c10*q1*dx - dt*dot(grad(mu1),grad(q1))*dx
L1 = mu1*q2*dx + epsu**2*dot(grad(c1),grad(q2))*dx - f1*q2*dx
L2 = c2*q3*dx - c20*q3*dx - dt*dot(grad(mu2),grad(q3))*dx + dt*sigma*(c2-vbar)*q3*dx
L3 = mu2*q4*dx + epsv**2*dot(grad(c2),grad(q4))*dx - f2*q4*dx
L = L0 + L1 + L2 + L3


# Output file
file_results = XDMFFile(f"./BCP/coupled-ch.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True

def save_to_file(uu, t):
    _c1,  _mu1, _c2, _mu2 = uu.split()
    _c1.rename("u", "u field")
    _c2.rename("v", "v field")
    _mu1.rename("mu_u", "chemical potential u")
    _mu2.rename("mu_v", "chemical potential v")
    file_results.write(_c1, t)
    file_results.write(_mu1, t)
    file_results.write(_c2, t)
    file_results.write(_mu2, t)

count = 0
for t in time_array:
    if t < 0.01:
        save_to_file(uu, t)
    elif count%10==0:
        save_to_file(uu, t)

    solve(L==0, uu  ,solver_parameters={"newton_solver":{"linear_solver":"gmres",
                                                              "relative_tolerance": 1e-6,
                                                              #"preconditioner":"ilu",
                                                              "convergence_criterion":"incremental",
                                                              }})
    uu0.vector()[:] = uu.vector()
    if count%100 ==0:
        print(f"Time = {t}")
    count += 1
