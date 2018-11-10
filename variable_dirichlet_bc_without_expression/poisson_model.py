""" Implement a Poisson solver with FEniCS and verifies it via MMS. """
import fenics
import dolfin
import math
import sys


class PoissonModel:
    
    def __init__(self, mesh, boundary_condition_value, source_term):
        
        element = fenics.FiniteElement('P', mesh.ufl_cell(), 1)
        
        function_space = fenics.FunctionSpace(mesh, element)
        
        solution = fenics.Function(function_space)
        
        
        class Boundaries(fenics.SubDomain):

            def inside(self, x, on_boundary):

                return on_boundary
                
        boundaries = Boundaries()
        
        boundary_condition = fenics.DirichletBC(
            function_space, boundary_condition_value, boundaries)
        
        
        dot, grad = fenics.dot, fenics.grad
        
        dx = fenics.dx
        
        f = source_term
        
        u = fenics.TrialFunction(function_space)
        
        v = fenics.TestFunction(function_space)
        
        problem = fenics.LinearVariationalProblem(
            a = dot(grad(v), grad(u))*dx,
            L = v*f*dx,
            u = solution,
            bcs = boundary_condition)
        
        solver = fenics.LinearVariationalSolver(problem)
        
        
        self.solution = solution
        
        self.solver = solver
        
        """ `solver` breaks (presumably due to some kind of scoping issue)
        if we don't make `boundary_condition` an attribute. 
        """
        self.boundary_condition = boundary_condition


def manufactured_solution(mesh):
    
    sin, pi = fenics.sin, fenics.pi
    
    x, y = fenics.SpatialCoordinate(mesh)
    
    return sin(2.*pi*x)*sin(pi*y)

    
def source_term(manufactured_solution):
    """ Derive the source terms which will yield the manufactured solution
        by substituting the manufactured solution into the strong form
        
            $ - \nabla \cdot \left( \nabla u \right) = f $
    """
    div, grad = fenics.div, fenics.grad
    
    u = manufactured_solution
    
    f = -div(grad(u))
    
    return f


def L2_error(manufactured_solution, computed_solution):
    
    u_M = manufactured_solution
    
    u_h = computed_solution
    
    return math.sqrt(fenics.assemble((u_h - u_M)**2*fenics.dx))


def test__verify_convergence_order_via_MMS(
        grid_sizes = (8, 16, 32), 
        expected_order = 2, 
        tolerance = 0.1):
    
    L2_errors = []

    for M in grid_sizes:
        
        mesh = fenics.UnitSquareMesh(M, M)
        
        model = PoissonModel(
            mesh = mesh,
            boundary_condition_value = manufactured_solution(mesh),
            source_term = source_term(manufactured_solution(mesh)))
        
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
    
    test__verify_convergence_order_via_MMS()
    