#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 13:14:32 2020

@author: bilginkocak
"""
from dolfin import *
import numpy as np
import random
# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["quadrature_degree"] = 5 #

mesh = BoxMesh(Point(0,0,0),Point(0.5,0.5,0.5),30,30,30)

V = VectorElement("Lagrange", mesh.ufl_cell(),degree=1)
P = FiniteElement("Lagrange", mesh.ufl_cell(),degree=1)
#TH = V * P * P
TH = MixedElement([V,P,P])
W = FunctionSpace(mesh, TH)

duu = TrialFunction(W)            # Incremental displacement
u_t, c_t, mu_t = TestFunctions(W)
uu = Function(W)
uu0 = Function(W)
uu00 = Function(W)
u, c ,mu = split(uu)
u0, c0, _, = split(uu0)
u00, c00, _, = split(uu00)

d = len(u)
dt = 0.1
tend = 1000
num = int(tend/dt)
time = np.linspace(0,tend, num)
gamma = Constant(3)
nu = Constant(0.3)
beta = Constant(2*nu/(1-2*nu))
omega = Constant(1)
A = Constant(10)
B = Constant(25)
epsilon = Constant(0.001)
M = Constant(0.001)

# Class representing the intial conditions
class InitialConditions(UserExpression):
    def __init__(self, **kwargs):
        random.seed(2 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)
    def eval(self, values, x):
        values[0] = 0.0
        values[1] = 0.0
        values[2] = 0.0
        values[3] = (random.uniform(0, 1))
        values[4] = 0.0
    def value_shape(self):
        return (5,)
# Create intial conditions and interpolate
u_init = InitialConditions(degree=1)
uu.interpolate(u_init)
uu0.interpolate(u_init)
uu00.interpolate(u_init)

Jc = 1 + omega*(c-c00)
I = Identity(d)
F = I + grad(u)
Fn = I + grad(u0)
C = F.T*F
Cn = Fn.T*Fn
Fe = pow(Jc,-1/3)*F
Fne = pow(Jc,-1/3)*Fn

dpsi_el_dFe = gamma*(Fe - pow(det(Fe), -beta)*inv(Fe).T)
dpsi_dF = pow(Jc, -1/3)*dpsi_el_dFe
p = -1/3*pow(Jc, -1/3)*inner(dpsi_el_dFe, F)
dpsi_dc = A*ln(c/(1-c)) + B*(1 - 2*c) + omega*p/Jc  #-1/3*pow(Jc,-4/3)*omega*inner(dpsi_el_dFe,F)#
dpsi_dgradc = epsilon*grad(c)
dphi_dM = c0*(1-c0)*M*inv(Cn)*(-grad(mu))

P = dpsi_dF
g = mu - dpsi_dc
K = dpsi_dgradc
H = dphi_dM

f1 = inner(P, grad(u_t))*dx
f2 = (c-c0)*c_t*dx - dt*dot(H, grad(c_t))*dx
f3 = g*mu_t*dx - dot(K, grad(mu_t))*dx
F =  f1 + f2 + f3

file_u = File('./miehe4/outputs/u.pvd')
file_c = File('./miehe4/outputs/c.pvd')
file_mu = File('./miehe4/outputs/mu.pvd')


count = 0
for t in time:

    if t <= 10:
      _u, _c, _mu = uu.split()
      file_u << (_u, t)
      file_c << (_c, t)
      file_mu << (_mu, t)
    else:
      if count % 100 == 0:
        _u, _c, _mu = uu.split()
        file_u << (_u, t)
        file_c << (_c, t)
        file_mu << (_mu, t)
    print("time = " + str(t))
    solve(F==0, uu, solver_parameters={"newton_solver":{"linear_solver":"gmres",
                                                                        #"relative_tolerance": 1e-6,
                                                                        #"preconditioner":"ilu",
                                                                        #"convergence_criterion":"incremental",
                                                                        }})

    count += 1

    uu0.assign(uu)
