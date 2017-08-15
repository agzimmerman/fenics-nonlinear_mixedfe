import fenics
import manual_newton


''' @brief Example of nonlinear problem with mixed finite elements

@detail This is the well-known steady lid-driven cavity problem,
modeled by the incompressible Navier-Stokes mass and momentum equations.

@todo Debug the Jacobian bilinear variational form.'''

def nonlinear_mixedfe(automatic_jacobian=True):


    # Set numerical parameters.
    mesh = fenics.UnitSquareMesh(10, 10, 'crossed')
    
    gamma = 1.e-7
    
    pressure_degree = 1


    # Set function spaces for the variational form     .
    velocity_degree = pressure_degree + 1
    
    velocity_space = fenics.VectorFunctionSpace(mesh, 'P', velocity_degree)

    pressure_space = fenics.FunctionSpace(mesh, 'P', pressure_degree) # @todo mixing up test function space

    ''' MixedFunctionSpace used to be available but is now deprecated. 
    To create the mixed space, I'm using the approach from https://fenicsproject.org/qa/11983/mixedfunctionspace-in-2016-2-0 '''
    velocity_element = fenics.VectorElement('P', mesh.ufl_cell(), velocity_degree)

    pressure_element = fenics.FiniteElement('P', mesh.ufl_cell(), pressure_degree)

    W_ele = fenics.MixedElement([velocity_element, pressure_element])

    W = fenics.FunctionSpace(mesh, W_ele)  
    
    
    # Set boundary conditions.
    bcs = [
        fenics.DirichletBC(W.sub(0), fenics.Expression(("1.", "0."), degree=velocity_degree + 1),
            'near(x[1],  1.)', method='topological'),
        fenics.DirichletBC(W.sub(0), fenics.Expression(("0.", "0."), degree=velocity_degree + 1),
            'near(x[0],  0.) | near(x[0],  1.) | near(x[1],  0.)', method='topological'),
        fenics.DirichletBC(W.sub(1), fenics.Expression("0.", degree=pressure_degree + 1),
            'near(x[0], 0.) && near(x[1], 0.)', method='pointwise')]    
    
    
    # Set nonlinear variational form.
    inner, dot, grad, div, sym = fenics.inner, fenics.dot, fenics.grad, fenics.div, fenics.sym
    
    D = lambda u : sym(grad(u))
    
    a = lambda u, v : 2.*inner(D(u), D(v))
    
    b = lambda u, q : -div(u)*q
    
    c = lambda w, z, v : dot(dot(grad(z), w), v)
    
    w = fenics.Function(W)
    
    u, p = fenics.split(w)

    v, q = fenics.TestFunctions(W)

    F = (b(u, q) - gamma*p*q + c(u, u, v) + a(u, v) + b(v, p))*fenics.dx
    
    
    # Solve nonlinear problem.
    if automatic_jacobian:
        
        JF = fenics.derivative(F, w)
        
    else:
        
        w_k = fenics.Function(W)
        
        u_k, p_k = fenics.split(w_k)
        
        dw = fenics.TrialFunction(W)
        
        du, dp = fenics.split(dw)
        
        A = (b(du, q) - gamma*dp*q + c(du, u_k, v) + c(u_k, du, v) + a(du, v) + b(v, dp))*fenics.dx

        L = (b(u_k, q) - gamma*p_k*q + c(u_k, u_k, v) + a(u_k, v) + b(v, p_k))*fenics.dx

        JF = A - L
    
    problem = fenics.NonlinearVariationalProblem(F, w, bcs, JF)

    solver  = fenics.NonlinearVariationalSolver(problem)

    solver.solve()



if __name__=='__main__':
    
    nonlinear_mixedfe(automatic_jacobian=True)
    
    nonlinear_mixedfe(automatic_jacobian=False)
    