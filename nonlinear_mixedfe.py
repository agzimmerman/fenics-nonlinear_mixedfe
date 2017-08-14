import fenics


''' @brief Example of nonlinear problem with mixed finite elements

@detail This is the well-known steady lid-driven cavity problem,
modeled by the incompressible Navier-Stokes mass and momentum equations.

@todo Debug the Jacobian bilinear variational form.'''

def nonlinear_mixedfe(automatic_jacobian):

    # Physical parameters
    mu = 1.

    # Set numerical parameters.
    m = 20

    mesh = fenics.UnitSquareMesh(m, m, 'crossed')
    
    gamma = 1.e-7
    
    pressure_degree = 1


    # Set function spaces for the variational form     .
    velocity_degree = pressure_degree + 1
    
    velocity_space = fenics.VectorFunctionSpace(mesh, 'P', velocity_degree)

    pressure_space = fenics.FunctionSpace(mesh, 'P', pressure_degree) # @todo mixing up test function space

    ''' MixedFunctionSpace used to be available but is now deprecated. 
    The way that fenics separates function spaces and elements is confusing.
    To create the mixed space, I'm using the approach from https://fenicsproject.org/qa/11983/mixedfunctionspace-in-2016-2-0 '''
    velocity_element = fenics.VectorElement('P', mesh.ufl_cell(), velocity_degree)

    pressure_element = fenics.FiniteElement('P', mesh.ufl_cell(), pressure_degree)

    W_ele = fenics.MixedElement([velocity_element, pressure_element])

    W = fenics.FunctionSpace(mesh, W_ele)  

                
                
    # Set boundary conditions.
    lid = 'near(x[1],  1.)'

    fixed_walls = 'near(x[0],  0.) | near(x[0],  1.) | near(x[1],  0.)'

    bottom_left_corner = 'near(x[0], 0.) && near(x[1], 0.)'

    bcs = [
        fenics.DirichletBC(W.sub(0), fenics.Expression(("1.", "0."), degree=velocity_degree + 1), lid, method='topological'),
        fenics.DirichletBC(W.sub(0), fenics.Expression(("0.", "0."), degree=velocity_degree + 1), fixed_walls, method='topological'),
        fenics.DirichletBC(W.sub(1), fenics.Expression("0.", degree=pressure_degree + 1), bottom_left_corner, method='pointwise')]    

    
    # Set nonlinear variational form.
    inner, dot, grad, div, sym = fenics.inner, fenics.dot, fenics.grad, fenics.div, fenics.sym

    def a(mu, u, v):

        def D(u):
        
            return sym(grad(u))
        
        return 2.*mu*inner(D(u), D(v))


    def b(u, q):
        
        return -div(u)*q
        

    def c(w, z, v):
       
        return dot(dot(grad(z), w), v)
        
    
    w = fenics.TrialFunction(W)
            
    u, p = fenics.split(w)

    v, q = fenics.TestFunctions(W)

    F = (
        b(u, q) - gamma*p*q
        + c(u, u, v) + a(mu, u, v) + b(v, p)
        )*fenics.dx
        
        
    # Set Jacobian bilinear variational form.
    w_k = fenics.Function(W)
    
    R = fenics.action(F, w_k)

    if automatic_jacobian:

        JR = fenics.derivative(R, w_k)
        
    else:

        u_k, p_k = fenics.split(w_k)

        dw = fenics.Function(W)

        du, dp = fenics.split(dw)

        JR = (
            b(du, q) - gamma*dp*q
            - (b(u_k, q) - gamma*p_k*q)
            + c(du, u_k, v) + c(u_k, du, v) + a(mu, du, v) + b(v, dp)
            + c(u_k, u_k, v) + a(mu, u_k, v) + b(v, p_k)
            )*fenics.dx
    

    # Solve the problem.
    problem = fenics.NonlinearVariationalProblem(R, w_k, bcs, JR)
    
    solver  = fenics.NonlinearVariationalSolver(problem)
    
    solver.solve()
    
        
if __name__=='__main__':

    nonlinear_mixedfe(automatic_jacobian=True)
    
    nonlinear_mixedfe(automatic_jacobian=False)