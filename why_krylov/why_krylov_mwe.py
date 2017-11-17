""" This script implements a Poisson solver using FEniCS."""
import fenics


def boundary(x, on_boundary):

    return on_boundary
    
    
mesh = fenics.UnitSquareMesh(8, 8)

x = fenics.SpatialCoordinate(mesh)

exact_solution = 1. + x[0]**2 + 2.*x[1]**2

f = -6.

V = fenics.FunctionSpace(mesh, 'P', 2)
    
bc = fenics.DirichletBC(V, exact_solution, boundary)

u = fenics.TrialFunction(V)

v = fenics.TestFunction(V)

dot, grad = fenics.dot, fenics.grad

a = dot(grad(v), grad(u))*fenics.dx

L = v*f*fenics.dx

solution = fenics.Function(V)

fenics.solve(a == L, solution, bc)
