""" This module tests melting_pcm.py on the 1D Stefan problem."""
import fenics
import melting_pcm
import scipy.optimize as opt

def verify_pci_position(true_pci_position, solution, tolerance):

    def theta(x):
        
        values = solution.leaf_node()(fenics.Point(x))
        
        return values[2]
    
    pci_pos = opt.newton(theta, 0.1)
    
    assert(abs(pci_pos - true_pci_position) < tolerance)
    
    
def refine_near_left_boundary(mesh, cycles):
    """ Refine mesh near the left boundary.
    The usual approach of using SubDomain and EdgeFunction isn't appearing to work
    in 1D, so I'm going to just loop through the cells of the mesh and set markers manually.
    """
    for i in range(cycles):
        
        cell_markers = fenics.CellFunction("bool", mesh)
        
        cell_markers.set_all(False)
        
        for cell in fenics.cells(mesh):
            
            found_left_boundary = False
            
            for vertex in fenics.vertices(cell):
                
                if fenics.near(vertex.x(0), 0.):
                    
                    found_left_boundary = True
                    
            if found_left_boundary:
                
                cell_markers[cell] = True
                
                break # There should only be one such point.
                
        mesh = fenics.refine(mesh, cell_markers)
        
    return mesh

    
def run(automatic_jacobian,
        stefan_number,
        regularization_central_temperature,
        regularization_smoothing_parameter,
        time_step_size):
    """ Run the melting PCM problem."""
    
    
    # Make the mesh.
    coarse_mesh_size = 1
    
    mesh = fenics.UnitIntervalMesh(coarse_mesh_size)
    
    initial_hot_wall_refinement_cycles = 10
    
    mesh = refine_near_left_boundary(mesh, initial_hot_wall_refinement_cycles)

        
    # Run the solver.
    element = melting_pcm.mixed_element(mesh.ufl_cell())
    
    function_space = fenics.FunctionSpace(mesh, element)
    
    hot_wall, cold_wall, walls = "near(x[0],  0.)", "near(x[0],  1.)", \
        "near(x[0],  0.) | near(x[0],  1.)"
        
    T_h, T_c = 1., -1.
    
    solution = melting_pcm.solve(W = function_space,
        prandtl_number = 1.,
        stefan_number = stefan_number,
        gravity = [0.],
        w_n = fenics.interpolate(
            fenics.Expression(
                ("0.", "0.",
                    "(" + str(T_h) + " - " + str(T_c) + ")*near(x[0], 0.) +  " + str(T_c)), 
                element=element),
            function_space),
        bcs = [
            fenics.DirichletBC(function_space.sub(0), [0.], walls),
            fenics.DirichletBC(function_space.sub(2), T_h, hot_wall),
            fenics.DirichletBC(function_space.sub(2), T_c, cold_wall)],
        time_step_size = time_step_size,
        regularization_smoothing_parameter = regularization_smoothing_parameter,
        regularization_central_temperature = regularization_central_temperature,
        end_time = 0.01,
        adaptive_goal_tolerance = 1.e-8,
        automatic_jacobian = automatic_jacobian)
    
    return solution
    
    
def test_stefan_problem_Ste1():

    solution = run(stefan_number = 1.,
        time_step_size = 1.e-4,
        regularization_central_temperature = 0.,
        regularization_smoothing_parameter = 0.01,
        automatic_jacobian = False)

    verify_pci_position(solution=solution, true_pci_position = 0.076, tolerance = 1.e-3)
    
    
def test_stefan_problem_Ste1_autoJ():

    solution = run(stefan_number = 1.,
        time_step_size = 1.e-4,
        regularization_central_temperature = 0.,
        regularization_smoothing_parameter = 0.01,
        automatic_jacobian = True)   
    
    verify_pci_position(solution=solution, true_pci_position = 0.076, tolerance = 1.e-3)
    
    
if __name__=="__main__":

    test_stefan_problem_Ste1_pcm()
    
    test_stefan_problem_Ste1_autoJ()
    