""" This module tests navier_stokes.py"""
import fenics
import melting_pcm


def verify_against_wang2010(solution):

    data = {"Ra": 1.e6, "Pr": 0.71, "x": 0.5, 
        "y": [0., 0.15, 0.35, 0.5, 0.65, 0.85, 1.], 
        "ux": [0.0000, -0.065, -0.019, 0.0000, 0.019, 0.065, 0.0000]}
    
    print("ux, true_ux, absolute_error")
    
    for i, true_ux in enumerate(data["ux"]):
    
        p = fenics.Point(data["x"], data["y"][i])
        
        values = solution.leaf_node()(p)
        
        ux = values[0]*data["Pr"]/data["Ra"]**0.5
        
        absolute_error = abs(ux - true_ux)
        
        print(str(ux) + ", " + str(true_ux) + ", " + str(absolute_error))
        
        assert(absolute_error < 1.e-3)
    
    print("Verified against wang2010.")


def run(automatic_jacobian = True):
    """ Run the melting PCM problem."""
    
    
    # Make the mesh.
    coarse_mesh_size = 4
    
    mesh = fenics.UnitSquareMesh(coarse_mesh_size, coarse_mesh_size)

    
    # Run the solver.
    element = melting_pcm.mixed_element(mesh.ufl_cell())
    
    function_space = fenics.FunctionSpace(mesh, element)
    
    hot_wall, cold_wall, walls = "near(x[0],  0.)", "near(x[0],  1.)", \
        "near(x[0],  0.) | near(x[0],  1.) | near(x[1],  0.) | near(x[1], 1.)"
        
    T_h, T_c = 0.5, -0.5
    
    solution = melting_pcm.solve(W = function_space,
        w_n = fenics.interpolate(
            fenics.Expression(
                ("0.", "0.", "0.", 
                    str(T_h) + "*" + hot_wall + " " + str(T_c) + "*" + cold_wall), 
                element=element),
            function_space),
        bcs = [
            fenics.DirichletBC(function_space.sub(0), (0., 0.), walls),
            fenics.DirichletBC(function_space.sub(2), T_h, hot_wall),
            fenics.DirichletBC(function_space.sub(2), T_c, cold_wall)],
        regularization_central_temperature = -1.,
        time_step_size = 1.e-3,
        time_step_geometric_growth_factor = 2.,
        adaptive_goal = "horizontal_velocity",
        adaptive_goal_tolerance = 1.e-5,
        end_time = 1.e8,
        automatic_jacobian = automatic_jacobian)
    
    verify_against_wang2010(solution)
    
    
def test_natural_convection():

    run(automatic_jacobian = False)

    
def test_natural_convection_autoJ():

    run(automatic_jacobian = True)    
    
    
if __name__=="__main__":

    test_natural_convection()
    
    test_natural_convection_autoJ()
    