""" This script implements a Poisson solver using FEniCS."""
import fenics


def boundary(x, on_boundary):

    return on_boundary
    
    
mesh = fenics.UnitSquareMesh(8, 8)

x = fenics.SpatialCoordinate(mesh)

V = fenics.FunctionSpace(mesh, 'P', 2)

u, v = fenics.TrialFunction(V), fenics.TestFunction(V)

fenics.set_log_level(fenics.DEBUG)

dot, grad = fenics.dot, fenics.grad

fenics.solve(dot(grad(v), grad(u))*fenics.dx == v*-6.*fenics.dx,
    fenics.Function(V),
    fenics.DirichletBC(V, 1. + x[0]**2 + 2.*x[1]**2, boundary))
