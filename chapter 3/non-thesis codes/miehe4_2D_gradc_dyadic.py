#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 22:12:33 2020

added term C:outer(grad(c),grad(c))
@author: bilginkocak
"""
from dolfin import *
import numpy as np
import random
import time
start_time = time.time()
mesh = RectangleMesh(Point(0, 0), Point(0.5,0.5),96,96)
V = VectorElement("Lagrange", mesh.ufl_cell(),degree=1)
P = FiniteElement("Lagrange", mesh.ufl_cell(),degree=1)
#TH = V * P * P
TH = MixedElement([V,P,P])
W = FunctionSpace(mesh, TH)
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["quadrature_degree"] = 5

duu = TrialFunction(W)            # Incremental displacement
u_t, c_t, mu_t = TestFunctions(W)
uu = Function(W)
uu0 = Function(W)
uu00 = Function(W)
u, c ,mu = split(uu)
u0, c0, _, = split(uu0)
u00, c00, _, = split(uu00)

dt = 0.1
tend = 1300
num = int(tend/dt) + 1
time_array = np.linspace(0,tend, num)
d = len(u)
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
        values[2] = (random.uniform(0, 1))
        values[3] = 0.0
    def value_shape(self):
        return (4,)
# Create intial conditions and interpolate
u_init = InitialConditions(degree=1)
uu.interpolate(u_init)
uu0.interpolate(u_init)
uu00.interpolate(u_init)

def leftbottom(x, on_boundary):
  return near(x[0],0) and near(x[1],0)
def lefttop(x, on_boundary):
  return near(x[0],0) and near(x[1],0.5)
bc_lb = DirichletBC(W.sub(0), Constant((0,0)), leftbottom, method='pointwise')
bc_lt = DirichletBC(W.sub(0).sub(0), Constant(0), lefttop, method='pointwise')
bcs = [bc_lb, bc_lt]

I = Identity(d)
Jc = 1 + omega*(c-c00)
F = I + grad(u)
Fn = I + grad(u0)
C = F.T*F
Cn = Fn.T*Fn
Fe = pow(Jc, -1/3)*F
Fne = pow(Jc, -1/3)*Fn

# New terms
c = variable(c)
gradc = grad(c)
gradc = variable(gradc)
Fe = variable(Fe)
added_term = pow(1 + omega*(c-c00), -2/3)*inner(F.T*F,outer(gradc,gradc))
added_term_dgradc = diff(added_term, gradc)
added_term_dFe = diff(added_term, Fe)
added_term_dc = diff(added_term, c)


dpsi_el_dFe = gamma*(Fe - pow(det(Fe), -beta)*inv(Fe).T) + added_term_dFe/1000
dpsi_dF = pow(Jc, -1/3)*dpsi_el_dFe
p = -1/3*pow(Jc, -1/3)*inner(dpsi_el_dFe, F)
dpsi_dc = A*ln(c/(1-c)) + B*(1 - 2*c) + omega*p/Jc + added_term_dc/1000
dpsi_dgradc = epsilon*grad(c) + added_term_dgradc/1000
dphi_dM = c0*(1-c0)*M*inv(Cn)*(-grad(mu))

P = dpsi_dF
g = mu - dpsi_dc
K = dpsi_dgradc
H = dphi_dM

f1 = inner(P, grad(u_t))*dx
f2 = (c-c0)*c_t*dx - dt*dot(H, grad(c_t))*dx
f3 = g*mu_t*dx - dot(K, grad(mu_t))*dx
F =  f1 + f2 + f3

file_results = XDMFFile("./miehe4_2d_dyadic/miehe4_2d.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True

# file_u = File('miehe4_2d_1/u.pvd')
# file_c = File('miehe4_2d_1/c.pvd')
# file_mu = File('miehe4_2d_1/mu.pvd')
_u, _c, _mu = uu.split()
_u.rename("u", "displacement")
_c.rename("c", "concentration")
_mu.rename("mu", "chemical potential")

# t = 0
# file_u << (_u, t)
# file_c << (_c, t)
# file_mu << (_mu, t)
#uu = Function(W)
def save_to_file(uu, t):
    _u, _c, _mu = uu.split()
    _u.rename("u", "displacement")
    _c.rename("c", "concentration")
    _mu.rename("mu", "chemical potential")
    file_results.write(_u, t)
    file_results.write(_c, t)
    file_results.write(_mu, t)

count = 0
for t in time_array:
    if t <= 1:
      save_to_file(uu, t)
    elif t<=10:
      if count % 10 == 0:
        save_to_file(uu, t)
    else:
      if count % 100 == 0:
        save_to_file(uu, t)
    solve(F==0, uu, bcs ,solver_parameters={"newton_solver":{"linear_solver":"lu",
                                                                        #"relative_tolerance": 1e-6,
                                                                        #"preconditioner":"ilu",
                                                                        #"convergence_criterion":"incremental",
                                                                        }})

    count += 1
    print(f"time = {t}")
    uu0.assign(uu)

save_to_file(uu, t)

elapsed_time = time.time() - start_time
print("Elapsed time : " + str(elapsed_time))
