""" This module tests navier_stokes.py"""
import fenics
import natural_convection
    

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
    
    
def test_natural_convection():
    """ Verify against the reference solution from wang2010."""
    coarse_mesh = fenics.UnitSquareMesh(2, 2)
    
    element = natural_convection.mixed_element(coarse_mesh.ufl_cell())
    
    function_space = fenics.FunctionSpace(coarse_mesh, element)
    
    hot_wall = "near(x[0],  0.)"
    
    cold_wall = "near(x[0],  1.)"
    
    walls = "near(x[0],  0.) | near(x[0],  1.) | near(x[1],  0.) | near(x[1], 1.)"
        
    T_h = 0.5
    
    T_c = -0.5
    
    initial_values = fenics.interpolate(
        fenics.Expression(
            ("0.", "0.", "0.", 
                str(T_h) + "*" + hot_wall + " " + str(T_c) + "*" + cold_wall), 
            element=element),
        function_space)
    
    solution = natural_convection.solve(W = function_space,
        w_n = initial_values,
        bcs = [
            fenics.DirichletBC(function_space.sub(0), (0., 0.), walls),
            fenics.DirichletBC(function_space.sub(2), T_h, hot_wall),
            fenics.DirichletBC(function_space.sub(2), T_c, cold_wall)])

    verify_against_wang2010(solution)
    
    
if __name__=="__main__":

    test_natural_convection()
    