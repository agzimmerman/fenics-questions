""" This module solves the Navier-Stokes equations using FEniCS."""
import fenics


MAXIMUM_TIME_STEPS = 100

GAMMA = 1.e-7

REYNOLDS_NUMBER = 1.

DYNAMIC_VISCOSITY = 1.

STEADY_TOLERANCE = 1.e-8

ADAPTIVE_GOAL_TOLERANCE = 1.e-5

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
        initial_time_step_size = 1.e-3,
        rayleigh_number = 1.e6,
        gravity = (0., -1.),
        prandtl_number = 0.71):
    """ Construct and solve the variational problem."""
    psi_u, psi_p, psi_T = fenics.TestFunctions(W)
        
    w = fenics.Function(W)
    
    u, p, T = fenics.split(w)
    
    u_n, p_n, T_n = fenics.split(w_n)
    
    inner, dot, grad, div, sym = fenics.inner, fenics.dot, \
        fenics.grad, fenics.div, fenics.sym
        
    time_step_size = initial_time_step_size
    
    Delta_t = fenics.Constant(time_step_size)
    
    Re = fenics.Constant(REYNOLDS_NUMBER)
    
    Ra = fenics.Constant(rayleigh_number)
    
    Pr = fenics.Constant(prandtl_number)
    
    g = fenics.Constant(gravity)
    
    gamma = fenics.Constant(GAMMA)
    
    mu = fenics.Constant(DYNAMIC_VISCOSITY)
    
    F = (-psi_p*div(u) - psi_p*gamma*p
        + 1./Delta_t*dot(psi_u, u - u_n) + dot(psi_u, dot(grad(u), u)) 
            + 2.*mu*inner(sym(grad(psi_u)), sym(grad(u))) - div(psi_u)*p
            + dot(psi_u, T*Ra/(Pr*Re**2)*g)
        + 1./Delta_t*psi_T*(T - T_n) + dot(grad(psi_T), 1./Pr*grad(T) - T*u)
        )*fenics.dx

    JF = jacobian = fenics.derivative(F, w, fenics.TrialFunction(W))
    
    problem = fenics.NonlinearVariationalProblem(F, w, bcs, JF)

    M = (Pr*u[0]/Ra**0.5)**2*fenics.dx
    
    solver = fenics.AdaptiveNonlinearVariationalSolver(problem, M)
    
    time = 0.
    
    with fenics.XDMFFile("output/solution.xdmf") as solution_file:
    
        write_solution(solution_file, w_n, time)
        
        for it in range(MAXIMUM_TIME_STEPS):

            solver.solve(ADAPTIVE_GOAL_TOLERANCE)
    
            time += time_step_size
            
            write_solution(solution_file, w, time)
            
            if steady(w, w_n):
            
                break
                
            w_n.leaf_node().vector()[:] = w.leaf_node().vector() # Reset initial values.
            
            
            # Double time step size to quickly reach steady state.
            time_step_size *= 2.
            
            Delta_t.assign(time_step_size) 

    return w
