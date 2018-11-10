import fenics


mesh = fenics.UnitIntervalMesh(3)

P1 = fenics.FiniteElement("P", mesh.ufl_cell(), 1)

element = fenics.MixedElement([P1, P1])

function_space = fenics.FunctionSpace(mesh, element)

x = fenics.SpatialCoordinate(mesh)

bc = fenics.DirichletBC(function_space.sub(0), 0., "on_boundary")

bc = fenics.DirichletBC(function_space.sub(0).collapse(), x[0], "on_boundary")

bc = fenics.DirichletBC(function_space.sub(0), x[0], "on_boundary")
