""" This module solves the Navier-Stokes equations using FEniCS."""
import fenics


GAMMA = 1.e-7

REYNOLDS_NUMBER = 1.

DYNAMIC_VISCOSITY = 1.

TIME_EPSILON = 1.e-8

LIQUID_VISCOSITY = 1.

MAX_NEWTON_ITERATIONS = 50

NEWTON_RELAXATION_PARAMETER = 1.

STEADY_TOLERANCE = 1.e-4

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

    
def steady(w, w_n):
    """Check if solution has reached an approximately steady state."""
    steady = False
    
    time_residual = fenics.Function(w.function_space().leaf_node())
    
    time_residual.assign(w.leaf_node() - w_n.leaf_node())
    
    unsteadiness = fenics.norm(time_residual, "L2")/fenics.norm(w_n.leaf_node(), "L2")
    
    print("Unsteadiness (L2 norm of relative time residual), || w_{n+1} || / || w_n || = "
        + str(unsteadiness))

    if (unsteadiness < STEADY_TOLERANCE):
        
        steady = True
        
        print("Reached steady state.")
    
    return steady
    
    
def solve(W, w_n, bcs, 
        time_step_size = 1.e-3,
        end_time = 0.02,
        rayleigh_number = 1.e6,
        stefan_number = 1.,
        prandtl_number = 0.71,
        solid_viscosity = 1.e-4,
        regularization_central_temperature = 0.1,
        regularization_smoothing_factor = 0.025,
        gravity = (0., -1.),
        time_step_geometric_growth_factor = 1.,
        adaptive_goal = "phase",
        adaptive_goal_tolerance = 1.e-4,
        automatic_jacobian = True):
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
    
    Ste = fenics.Constant(stefan_number)
    
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

        
    def b(u, q):
    
        return -div(u)*q

    
    def D(u):
    
        return sym(grad(u))  # Symmetric part of velocity gradient
    
    
    def a(mu, u, v):
        
        return 2.*mu*inner(D(u), D(v))  # Stokes stress-strain
    
    
    def c(w, z, v):
        
        return dot(dot(grad(z), w), v) # Convection of the velocity field
    
    
    def f_B(T):
    
        return T*Ra/(Pr*Re**2)*g
    
    F = (
        b(u, psi_p) - psi_p*gamma*p
        + 1./Delta_t*dot(psi_u, u - u_n) + c(u, u, psi_u) + a(mu(T), u, psi_u) + b(psi_u, p)
        + dot(psi_u, f_B(T))
        + 1./Delta_t*psi_T*(T - T_n) + dot(grad(psi_T), 1./Pr*grad(T) - T*u)
        - 1./(Ste*Delta_t)*psi_T*(phi(T) - phi(T_n))
        )*fenics.dx

        
    delta_w = fenics.TrialFunction(W)
    
    if automatic_jacobian:
    
        JF = fenics.derivative(F, w, delta_w)
        
    else:
    
        delta_u, delta_p, delta_T = fenics.split(delta_w)
        
        def sech(theta):
    
            return 1./fenics.cosh(theta)
    
    
        def dphi(T):
    
            return -sech((T_f - T)/r)**2/(2.*r)
        
    
        def dP(T, P_L, P_S):
        
            return (P_S - P_L)*dphi(T)
            
            
        def dmu(T):
    
            return dP(T, mu_L, mu_S)
        
        
        def df_B(T):
    
            return Ra/(Pr*Re**2)*g
        
        
        JF = (
            b(delta_u, psi_p) - gamma*psi_p*delta_p
            + 1./Delta_t*dot(psi_u, delta_u)
            + c(u, delta_u, psi_u) + c(delta_u, u, psi_u)
            + b(psi_u, delta_p)
            + a(delta_T*dmu(T), u, psi_u) + a(mu(T), delta_u, psi_u)
            + dot(psi_u, delta_T*df_B(T))
            + 1./Delta_t*psi_T*delta_T
            - dot(grad(psi_T), T*delta_u)
            - dot(grad(psi_T), delta_T*u)
            + 1./Pr*dot(grad(psi_T), grad(delta_T))
            - 1./(Ste*Delta_t)*psi_T*delta_T*dphi(T)
            )*fenics.dx
        
    problem = fenics.NonlinearVariationalProblem(F, w, bcs, JF)

    if adaptive_goal == "phase":
    
        M = phi(T)*fenics.dx
        
    elif adaptive_goal == "horizontal_velocity":
    
        M = (Pr/Ra**0.5*u[0])**2*fenics.dx 
        
    else:
    
        assert(False)
    
    solver = fenics.AdaptiveNonlinearVariationalSolver(problem, M)
    
    solver.parameters["nonlinear_variational_solver"]["newton_solver"]\
        ["maximum_iterations"] = MAX_NEWTON_ITERATIONS
        
    solver.parameters["nonlinear_variational_solver"]["newton_solver"]\
        ["relaxation_parameter"] = NEWTON_RELAXATION_PARAMETER
    
    time = 0.
    
    with fenics.XDMFFile("output/solution.xdmf") as solution_file:
    
        write_solution(solution_file, w_n, time)
        
        while (time < (end_time - TIME_EPSILON)):

            solver.solve(adaptive_goal_tolerance)
    
            time += time_step_size
            
            print("Current time / end time = " + str(time) + " / " + str(end_time))
            
            write_solution(solution_file, w, time)
            
            if steady(w, w_n):
            
                break
            
            w_n.leaf_node().vector()[:] = w.leaf_node().vector() # Reset initial values.
            
            
            # Update the time step size.
            time_step_size *= time_step_geometric_growth_factor
            
            Delta_t.assign(time_step_size)

    return w
