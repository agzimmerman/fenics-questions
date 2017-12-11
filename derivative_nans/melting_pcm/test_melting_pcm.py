""" This module tests navier_stokes.py"""
import fenics
import melting_pcm

    
def run(automatic_jacobian = True):
    """ Run the melting PCM problem."""
    
    
    # Make the mesh.
    coarse_mesh_size = 1
    
    mesh = fenics.UnitSquareMesh(coarse_mesh_size, coarse_mesh_size, "crossed")
    
    initial_hot_wall_refinement_cycles = 6
    
    class HotWall(fenics.SubDomain):
        
        def inside(self, x, on_boundary):
        
            return on_boundary and fenics.near(x[0], 0.)

            
    hot_wall = HotWall()
    
    for i in range(initial_hot_wall_refinement_cycles):
        
        edge_markers = fenics.EdgeFunction("bool", mesh)
        
        hot_wall.mark(edge_markers, True)

        fenics.adapt(mesh, edge_markers)
        
        mesh = mesh.child()

        
    # Run the solver.
    element = melting_pcm.mixed_element(mesh.ufl_cell())
    
    function_space = fenics.FunctionSpace(mesh, element)
    
    hot_wall, cold_wall, walls = "near(x[0],  0.)", "near(x[0],  1.)", \
        "near(x[0],  0.) | near(x[0],  1.) | near(x[1],  0.) | near(x[1], 1.)"
        
    T_h, T_c = 1., -0.1
    
    solution = melting_pcm.solve(W = function_space,
        w_n = fenics.interpolate(
            fenics.Expression(
                ("0.", "0.", "0.", 
                    "(" + str(T_h) + " - " + str(T_c) + ")*(x[0] < 0.001) +  " + str(T_c)), 
                element=element),
            function_space),
        bcs = [
            fenics.DirichletBC(function_space.sub(0), (0., 0.), walls),
            fenics.DirichletBC(function_space.sub(2), T_h, hot_wall),
            fenics.DirichletBC(function_space.sub(2), T_c, cold_wall)],
        time_step_size = 1.e-3,
        end_time = 0.02,
        rayleigh_number = 1.e6,
        stefan_number = 1.,
        prandtl_number = 0.71,
        solid_viscosity = 1.e-4,
        regularization_central_temperature = 0.1,
        regularization_smoothing_parameter = 0.025,
        adaptive_goal_tolerance = 1.e-4,
        automatic_jacobian = automatic_jacobian)
    
    
def test_melting_toy_pcm():

    run(automatic_jacobian = False)

    
def test_melting_toy_pcm_autoJ():

    run(automatic_jacobian = True)    
    
    
if __name__=="__main__":

    test_melting_toy_pcm()
    
    test_melting_toy_pcm_autoJ()
    