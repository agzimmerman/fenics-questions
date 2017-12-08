""" This module solves the Navier-Stokes equations using FEniCS."""
import fenics


GAMMA = 1.e-7

REYNOLDS_NUMBER = 1.

DYNAMIC_VISCOSITY = 1.

ADAPTIVE_GOAL_TOLERANCE = 1.e-4

TIME_EPSILON = 1.e-8

LIQUID_VISCOSITY = 1.

def mixed_element(ufl_cell):
    """ Make a mixed finite element, 
    with the Taylor-Hood element for pressure and velocity."""
    velocity_ele = fenics.VectorElement('P', ufl_cell, 2)

    pressure_ele = fenics.FiniteElement('P', ufl_cell, 1)
    
    temperature_ele = fenics.FiniteElement('P', ufl_cell, 1)

    return fenics.MixedElement([velocity_ele, pressure_ele, temperature_ele])
    
    
def write_solution(solution_file, w, time):
    """Write the solution to disk."""
    print("Writing solution to HDF5+XDMF")
    
    velocity, pressure, temperature = w.leaf_node().split()
    
    velocity.rename("u", "velocity")
    
    pressure.rename("p", "pressure")
    
    temperature.rename("T", "temperature")
    
    for var in [velocity, pressure, temperature]:
    
        solution_file.write(var, time)

    
def solve(W, w_n, bcs, 
        time_step_size = 1.e-3,
        end_time = 0.02,
        rayleigh_number = 1.e6,
        stefan_number = 1.,
        prandtl_number = 0.71,
        solid_viscosity = 1.e-4,
        regularization_central_temperature = 0.1,
        regularization_smoothing_factor = 0.025,
        gravity = (0., -1.)):
    """ Construct and solve the variational problem."""
    psi_u, psi_p, psi_T = fenics.TestFunctions(W)
        
    w = fenics.Function(W)
    
    u, p, T = fenics.split(w)
    
    u_n, p_n, T_n = fenics.split(w_n)
    
    inner, dot, grad, div, sym = fenics.inner, fenics.dot, \
        fenics.grad, fenics.div, fenics.sym
        
    Delta_t = fenics.Constant(time_step_size)
    
    Re = fenics.Constant(REYNOLDS_NUMBER)
    
    Ra = fenics.Constant(rayleigh_number)
    
    Pr = fenics.Constant(prandtl_number)
    
    g = fenics.Constant(gravity)
    
    gamma = fenics.Constant(GAMMA)
    
    g = fenics.Constant(gravity)

    T_f = fenics.Constant(regularization_central_temperature)
    
    r = fenics.Constant(regularization_smoothing_factor)
    
    def phi(T):
    
        return 0.5*(1. + fenics.tanh((T_f - T)/r)) # Regularized solid volume fraction.
        
    
    def P(T, P_L, P_S):
    
        return P_L + (P_S - P_L)*phi(T)
        
        
    mu_L = fenics.Constant(LIQUID_VISCOSITY)
    
    mu_S = fenics.Constant(solid_viscosity)
    
    def mu(T):
    
        return P(T, mu_L, mu_S)

    
    F = (-psi_p*div(u) - psi_p*gamma*p
        + 1./Delta_t*dot(psi_u, u - u_n) + dot(psi_u, dot(grad(u), u)) 
            + 2.*mu(T)*inner(sym(grad(psi_u)), sym(grad(u))) - div(psi_u)*p
            + dot(psi_u, T*Ra/(Pr*Re**2)*g)
        + 1./Delta_t*psi_T*(T - T_n) + dot(grad(psi_T), 1./Pr*grad(T) - T*u)
        + 1./Delta_t*psi_T*(phi(T) - phi(T_n))
        )*fenics.dx

    JF = jacobian = fenics.derivative(F, w, fenics.TrialFunction(W))
    
    problem = fenics.NonlinearVariationalProblem(F, w, bcs, JF)

    M = phi(T)*fenics.dx
    
    solver = fenics.AdaptiveNonlinearVariationalSolver(problem, M)
    
    solver.parameters["nonlinear_variational_solver"]["newton_solver"]\
        ["maximum_iterations"] = 10
    
    time = 0.
    
    with fenics.XDMFFile("output/solution.xdmf") as solution_file:
    
        write_solution(solution_file, w_n, time)
        
        while (time < (end_time - TIME_EPSILON)):

            solver.solve(ADAPTIVE_GOAL_TOLERANCE)
    
            time += time_step_size
            
            print("Current time / end time = " + str(time) + " / " + str(end_time))
            
            write_solution(solution_file, w, time)
            
            w_n.leaf_node().vector()[:] = w.leaf_node().vector() # Reset initial values.

    return w
