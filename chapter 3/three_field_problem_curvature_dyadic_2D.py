#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 22:12:33 2020
# curvature calculation
solver ile hesapla.
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

VV = FunctionSpace(mesh, "Lagrange", 1)
dof_coordinates = VV.tabulate_dof_coordinates()
xv, yv = np.meshgrid((np.unique(dof_coordinates[:,0])), (np.unique(dof_coordinates[:,1])))
index_coord = [[i, coord] for i, coord in enumerate(dof_coordinates)]

test_v = TestFunctions(VV)
_K = Function(VV)
_H = Function(VV)
_kmax = Function(VV)
_kmin = Function(VV)

file_results = XDMFFile("./miehe4_2d_curvature/miehe4_2d.xdmf")
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
def surface_curvature(X,Y,Z):
    """
    This function calculates surface curvature values

     Args:
        X : X position
        Y : Y position
        Z : Z position
    Returns:
        kmax : Maximum principal curvature
        kmin : Minimum principal curvature
        K : Gaussian curvature
        H : Mean curvature
    """

    (lr,lb)=X.shape

    #First Derivatives
    Xv,Xu=np.gradient(X)
    Yv,Yu=np.gradient(Y)
    Zv,Zu=np.gradient(Z)


    #Second Derivatives
    Xuv,Xuu=np.gradient(Xu)
    Yuv,Yuu=np.gradient(Yu)
    Zuv,Zuu=np.gradient(Zu)

    Xvv,Xuv=np.gradient(Xv)
    Yvv,Yuv=np.gradient(Yv)
    Zvv,Zuv=np.gradient(Zv)

    #2D to 1D conversion
    #Reshape to 1D vectors
    Xu=np.reshape(Xu,lr*lb)
    Yu=np.reshape(Yu,lr*lb)
    Zu=np.reshape(Zu,lr*lb)
    Xv=np.reshape(Xv,lr*lb)
    Yv=np.reshape(Yv,lr*lb)
    Zv=np.reshape(Zv,lr*lb)
    Xuu=np.reshape(Xuu,lr*lb)
    Yuu=np.reshape(Yuu,lr*lb)
    Zuu=np.reshape(Zuu,lr*lb)
    Xuv=np.reshape(Xuv,lr*lb)
    Yuv=np.reshape(Yuv,lr*lb)
    Zuv=np.reshape(Zuv,lr*lb)
    Xvv=np.reshape(Xvv,lr*lb)
    Yvv=np.reshape(Yvv,lr*lb)
    Zvv=np.reshape(Zvv,lr*lb)

    Xu=np.c_[Xu, Yu, Zu]
    Xv=np.c_[Xv, Yv, Zv]
    Xuu=np.c_[Xuu, Yuu, Zuu]
    Xuv=np.c_[Xuv, Yuv, Zuv]
    Xvv=np.c_[Xvv, Yvv, Zvv]

    # First fundamental Coeffecients of the surface (E,F,G)

    E=np.einsum('ij,ij->i', Xu, Xu)
    F=np.einsum('ij,ij->i', Xu, Xv)
    G=np.einsum('ij,ij->i', Xv, Xv)

    m=np.cross(Xu,Xv,axisa=1, axisb=1)
    p=np.sqrt(np.einsum('ij,ij->i', m, m))
    n=m/np.c_[p,p,p]
    # n is the normal
    # Second fundamental Coeffecients of the surface (L,M,N), (e,f,g)
    L= np.einsum('ij,ij->i', Xuu, n) #e
    M= np.einsum('ij,ij->i', Xuv, n) #f
    N= np.einsum('ij,ij->i', Xvv, n) #g

    # Alternative formula for gaussian curvature in wikipedia
    # K = det(second fundamental) / det(first fundamental)
    # Gaussian Curvature
    K=(L*N-M**2)/(E*G-F**2)
    K=np.reshape(K,lr*lb)

    # wiki trace of (second fundamental)(first fundamental inverse)
    # Mean Curvature
    H = ((E*N + G*L - 2*F*M)/((E*G - F**2)))/2
    H = np.reshape(H,lr*lb)

    # Principle Curvatures
    kmax = H + np.sqrt(H**2 - K)
    kmin = H - np.sqrt(H**2 - K)

    return kmax, kmin, K, H

def save_to_file(uu, t):
    _u, _c, _mu = uu.split()
    _u.rename("u", "displacement")
    _c.rename("c", "concentration")
    _mu.rename("mu", "chemical potential")
    file_results.write(_u, t)
    file_results.write(_c, t)
    file_results.write(_mu, t)

def calc_surf_curvatures(c):
    f_x = c.dx(0)
    f_y = c.dx(1)
    E = 1 + f_x**2
    F = f_x*f_y
    G = 1 + f_y**2
    length = sqrt(1 + f_x**2 + f_y**2)
    f_xx = f_x.dx(0)
    f_xy = f_x.dx(1)
    f_yy = f_y.dx(1)
    L = f_xx/length
    M = f_xy/length
    N = f_yy/length
    K = (L*N - M**2)/(E*G - F**2)
    H = (G*L - 2*F*M + E*N)/(2*(E*G - F**2))
    kmax = (H + sqrt(H**2 - 4*K))/2
    kmin = (H - sqrt(H**2 - 4*K))/2
    return K, H, kmax, kmin

def save_curvatures(c, t):
    z = np.zeros((xv.shape[0], xv.shape[1]))
    for i in range(xv.shape[0]):
        for j in range(xv.shape[1]):
            z[i,j] = c((xv[i,j],yv[i,j]))
    kmax, kmin, K, H = surface_curvature(xv,yv,z)
    lw = int(np.sqrt(len(kmax)))
    kmax = np.reshape(kmax, (-1, lw))
    kmin = np.reshape(kmin, (-1, lw))
    H = np.reshape(H, (-1, lw))
    K = np.reshape(K, (-1, lw))

    kmax_vec = np.zeros((dof_coordinates.shape[0]))
    kmin_vec = np.zeros((dof_coordinates.shape[0]))
    K_vec = np.zeros((dof_coordinates.shape[0]))
    H_vec = np.zeros((dof_coordinates.shape[0]))
    for i,coord in index_coord:
        kmax_vec[i] = kmax[np.where(xv[0]==coord[1])[0][0],np.where(yv[:,0]==coord[0])[0][0]]
        kmin_vec[i] = kmin[np.where(xv[0]==coord[1])[0][0],np.where(yv[:,0]==coord[0])[0][0]]
        K_vec[i] = K[np.where(xv[0]==coord[1])[0][0],np.where(yv[:,0]==coord[0])[0][0]]
        H_vec[i] = H[np.where(xv[0]==coord[1])[0][0],np.where(yv[:,0]==coord[0])[0][0]]
    _K.vector()[:] = K_vec
    _H.vector()[:] = H_vec
    _kmax.vector()[:] = kmax_vec
    _kmin.vector()[:] = kmin_vec
    # _K = project(K, VV)
    # _H = project(H, VV)
    # _kmax = project(kmax, VV)
    # _kmin = project(kmin, VV)
    # L = _K*test_v*dx - K*test_v*dx
    # solve(L==0, _K)
    # file << _K_, t
    # _K.project(K)
    _K.rename("K", "Gaussian Curvature")
    _H.rename("H", "Mean Curvature")
    _kmax.rename("kmax", "Maximum Curvature")
    _kmin.rename("kmin", "Minimum Curvature")
    file_results.write(_K, t)
    file_results.write(_H, t)
    file_results.write(_kmax, t)
    file_results.write(_kmin, t)

count = 0
for t in time_array:
    if t <= 1:
      save_to_file(uu, t)
      save_curvatures(split(uu)[1], t)
    elif t<=10:
      if count % 10 == 0:
        save_to_file(uu, t)
        save_curvatures(split(uu)[1], t)
    else:
      if count % 100 == 0:
        save_to_file(uu, t)
        save_curvatures(split(uu)[1], t)
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
