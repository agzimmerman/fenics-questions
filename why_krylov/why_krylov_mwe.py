""" This module implements a Poisson solver using FEniCS."""
import fenics

def boundary(x, on_boundary):
    """ Mark the boundary for applying BC's."""
    return on_boundary
        
        
def solve(mesh, u_D, f):
    """ Solve the Poisson problem."""
    V = fenics.FunctionSpace(mesh, 'P', 2)
        
    bc = fenics.DirichletBC(V, u_D, boundary)
    
    u = fenics.TrialFunction(V)
    
    v = fenics.TestFunction(V)
    
    dot, grad = fenics.dot, fenics.grad
    
    a = dot(grad(v), grad(u))*fenics.dx
    
    L = v*f*fenics.dx
    
    solution = fenics.Function(V)
    
    fenics.solve(a == L, solution, bc)
    
    return solution
    
    
def verify_with_mms():
    """ Verify via the method of manufactured solution."""
    exact_u = fenics.Expression(
        '1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)
    
    u = solve(
        mesh = fenics.UnitSquareMesh(8, 8),
        u_D = exact_u,
        f = fenics.Constant(-6.0))
    
    L2_error = fenics.errornorm(exact_u, u, 'L2')
    
    print("L2 error = " + str(L2_error))
    
    assert(L2_error < 10*fenics.DOLFIN_EPS)
    
    print("Successfully verified with MMS.")


if __name__=='__main__':

    verify_with_mms()
    