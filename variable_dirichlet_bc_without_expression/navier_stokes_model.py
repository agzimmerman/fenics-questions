""" Implement a Poisson solver with FEniCS and verifies it via MMS. """
import fenics
import dolfin
from ufl import unit_vector
import math
import sys


class NavierStokesModel:
    
    def __init__(self, mesh, boundary_condition_values, source_terms = (0., 0.)):
        
        element = fenics.MixedElement(
            fenics.VectorElement('P', mesh.ufl_cell(), 2),
            fenics.FiniteElement('P', mesh.ufl_cell(), 1))
        
        function_space = fenics.FunctionSpace(mesh, element)
        
        solution = fenics.Function(function_space)
        
        
        class Boundaries(fenics.SubDomain):

            def inside(self, x, on_boundary):

                return on_boundary
                
        boundaries = Boundaries()
        
        boundary_conditions = [
            fenics.DirichletBC(
                function_space.sub(i),
                boundary_condition_values[i],
                boundaries)
            for i, g in enumerate(boundary_condition_values)]
        
        
        inner, dot, grad, div, sym = \
            fenics.inner, fenics.dot, fenics.grad, fenics.div, fenics.sym
        
        s_u, s_p = source_terms
        
        u, p = fenics.split(solution)
        
        psi_u, psi_p = fenics.TestFunctions(function_space)
        
        dot(psi_u, s_u) + psi_p*s_p
        
        F = (dot(psi_u, dot(grad(u), u) - s_u)
            - div(psi_u)*p + 2.*inner(sym(grad(psi_u)), sym(grad(u)))
            + psi_p*(div(u) - s_p))*fenics.dx
            
        
        problem = fenics.NonlinearVariationalProblem(
            F = F,
            u = solution,
            bcs = boundary_conditions,
            J = fenics.derivative(F, solution))
        
        solver = fenics.NonlinearVariationalSolver(problem)
        
        
        self.solution = solution
        
        self.solver = solver
        
        """ `solver` breaks (presumably due to some kind of scoping issue)
        if we don't make `boundary_condition` an attribute. 
        """
        self.boundary_conditions = boundary_conditions


def manufactured_solution(mesh):
    
    sin, pi = fenics.sin, fenics.pi
    
    x, y = fenics.SpatialCoordinate(mesh)
    
    ihat, jhat = unit_vector(0, 2), unit_vector(1, 2)
    
    u_M = sin(2.*pi*x)*sin(pi*y)*ihat + sin(pi*x)*sin(2.*pi*y)*jhat
    
    p_M = -0.5*(u_M[0]**2 + u_M[1]**2)
    
    return u_M, p_M

    
def source_terms(manufactured_solution):
    
    grad, dot, div, sym = fenics.grad, fenics.dot, fenics.div, fenics.sym
    
    u_M, p_M = manufactured_solution
    
    """ Working symbolically with the vector-valued solution in `fenics` 
    seems to require invocation of the cartesian unit vectors, 
    rather than using tuples or lists. """
    ihat, jhat = unit_vector(0, 2), unit_vector(1, 2)
    
    _u_M = u_M[0]*ihat + u_M[1]*jhat
    
    s_u = grad(_u_M)*_u_M + grad(p_M) - 2.*div(sym(grad(_u_M)))
    
    s_p = div(_u_M)
    
    return s_u, s_p


def L2_error(manufactured_solution, computed_solution):
    
    u_M, p_M = manufactured_solution
    
    ihat, jhat = unit_vector(0, 2), unit_vector(1, 2)
    
    _u_M = u_M[0]*ihat + u_M[1]*jhat
    
    u_h, p_h = computed_solution.split()
    
    return math.sqrt(fenics.assemble((
        fenics.dot(u_h - _u_M, u_h - _u_M) + (p_h - p_M)**2)*fenics.dx))


def test__verify_convergence_order_via_MMS(
        grid_sizes = (8, 16, 32), 
        expected_order = 2, 
        tolerance = 0.1,
        bc_approach = "automatic"):
    
    L2_errors = []

    for M in grid_sizes:
        
        mesh = fenics.UnitSquareMesh(M, M)
        
        """ We want to use the manufactured solution as the bc values,
        but something is wrong with our use of `fenics.DirichletBC`.
        For now we manufactured a solution with vanishing boundary values.
        """
        if bc_approach == "automatic":
        
            bc_values = manufactured_solution(mesh)
            
        elif bc_approach == "manual":
        
            bc_values = ((0., 0.), 0.)
            
        model = NavierStokesModel(
            mesh = mesh,
            boundary_condition_values = bc_values,
            source_terms = source_terms(manufactured_solution(mesh)))
        
        model.solver.solve()
        
        L2_errors.append(
            L2_error(manufactured_solution(mesh), model.solution))
    
    edge_lengths = [1./float(M) for M in grid_sizes]

    e, h = L2_errors, edge_lengths

    log = math.log

    orders = [(log(e[i + 1]) - log(e[i]))/(log(h[i + 1]) - log(h[i]))
              for i in range(len(e) - 1)]
    
    print("Edge lengths = " + str(edge_lengths))
    
    print("L2 norm errors = " + str(L2_errors))
    
    print("Convergence orders = " + str(orders))
    
    assert(abs(orders[-1] - expected_order) < tolerance)


if __name__ == "__main__":

    print("Using Python " + sys.version)

    print("Using fenics-" + dolfin.__version__)
    
    test__verify_convergence_order_via_MMS(bc_approach = "manual")
    
    test__verify_convergence_order_via_MMS(bc_approach = "automatic")
    