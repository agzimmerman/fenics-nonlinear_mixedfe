""" Solve a nonlinear problem with mixed finite elements using FEniCS.

This is the well-known steady lid-driven cavity problem,
modeled by the incompressible Navier-Stokes mass and momentum equations,
stabilized with a pressure penalty formulation.

Most of the theory can be found in 

    @book{donea2003finite,
      title={Finite element methods for flow problems},
      author={Donea, Jean and Huerta, Antonio},
      year={2003},
      publisher={John Wiley \& Sons}
    }

For the FEniCS implementation, we use the approach from section 1.2.4 of the FEniCS Book,

    @book{logg2012automated,
      title={Automated solution of differential equations by the finite element method: The FEniCS book},
      author={Logg, Anders and Mardal, Kent-Andre and Wells, Garth},
      volume={84},
      year={2012},
      publisher={Springer Science \& Business Media}
    }
"""
import fenics


"""Verify the solution against results published in...

    @article{ghia1982high,
      title={High-Re solutions for incompressible flow using 
        the Navier-Stokes equations and a multigrid method},
      author={Ghia, UKNG and Ghia, Kirti N and Shin, CT},
      journal={Journal of computational physics},
      volume={48},
      number={3},
      pages={387--411},
      year={1982},
      publisher={Elsevier}
    }
"""
def verify_against_ghia1982(w, mesh):

    data = {'Re': 100, 'x': 0.5,
        'y': [1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5000, 0.4531, 
              0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000],
        'ux': [1.0000, 0.8412, 0.7887, 0.7372, 0.6872, 0.2315, 0.0033, -0.1364, -0.2058,
               -0.2109, -0.1566, -0.1015, -0.0643, -0.0478, -0.0419, -0.0372, 0.0000]}
    
    bbt = mesh.bounding_box_tree()
    
    for i, true_ux in enumerate(data['ux']):
    
        p = fenics.Point(data['x'], data['y'][i])
        
        if bbt.collides_entity(p):
        
            wval = w(p)
            
            ux = wval[0]
            
            assert(abs(ux - true_ux) < 2.e-2)

    print("Verified successfully against Ghia1982.")

def nonlinear_mixedfe(automatic_jacobian=True, Re=100.):

    # Set physical parameters
    mu = 1./Re
    
    
    # Set numerical parameters.
    mesh = fenics.UnitSquareMesh(20, 20, 'crossed')
    
    gamma = 1.e-7  # Parameter for pressure penalty formulation, should be on the order of 1.e-7-1.e-8
    
    pressure_degree = 1


    # Set function spaces for the variational form     .
    velocity_degree = pressure_degree + 1  # Higher order velocity element needed for stability (Donea, Huerta 2003)
    
    velocity_space = fenics.VectorFunctionSpace(mesh, 'P', velocity_degree)

    pressure_space = fenics.FunctionSpace(mesh, 'P', pressure_degree) # @todo mixing up test function space

    ''' MixedFunctionSpace used to be available but is now deprecated. 
    To create the mixed space, I'm using the approach from https://fenicsproject.org/qa/11983/mixedfunctionspace-in-2016-2-0 '''
    velocity_element = fenics.VectorElement('P', mesh.ufl_cell(), velocity_degree)

    pressure_element = fenics.FiniteElement('P', mesh.ufl_cell(), pressure_degree)

    W_ele = fenics.MixedElement([velocity_element, pressure_element])

    W = fenics.FunctionSpace(mesh, W_ele)  
    
    
    # Set Dirichlet boundary conditions.
    bcs = [
        fenics.DirichletBC(W.sub(0),
            fenics.Expression(("1.", "0."),
            degree=velocity_degree + 1),
            'near(x[1],  1.)',
            method='topological'),
        fenics.DirichletBC(W.sub(0),
            fenics.Expression(("0.", "0."),
            degree=velocity_degree + 1),
            'near(x[0],  0.) | near(x[0],  1.) | near(x[1],  0.)',
            method='topological'),
        fenics.DirichletBC(W.sub(1),
            fenics.Expression("0.",
            degree=pressure_degree + 1),
            'near(x[0], 0.) && near(x[1], 0.)',
            method='pointwise')]    
    
    
    # Set nonlinear variational form.
    inner, dot, grad, div, sym = fenics.inner, fenics.dot, fenics.grad, fenics.div, fenics.sym
    
    D = lambda u : sym(grad(u))
    
    a = lambda u, v : 2.*mu*inner(D(u), D(v))
    
    b = lambda u, q : -div(u)*q
    
    c = lambda w, z, v : dot(dot(grad(z), w), v)
    
    
    # Solve nonlinear problem.
    dw = fenics.TrialFunction(W)
    
    du, dp = fenics.split(dw)
    
    v, q = fenics.TestFunctions(W)
        
    w_ = fenics.Function(W)
    
    u_, p_ = fenics.split(w_)
    
    F = (
        b(u_, q) - gamma*p_*q 
        + c(u_, u_, v) + a(u_, v) + b(v, p_)
        )*fenics.dx
    
    if automatic_jacobian:
    
        JF = fenics.derivative(F, w_, dw)
        
    else:

        JF = (
            b(du, q) - gamma*dp*q
            + c(u_, du, v) + c(du, u_, v) + a(du, v) + b(v, dp)
            )*fenics.dx
    
    problem = fenics.NonlinearVariationalProblem(F, w_, bcs, JF)

    solver  = fenics.NonlinearVariationalSolver(problem)

    solver.solve()
    
    return w_, mesh

    
def test_nonlinear_mixedfe_automatic_jacobian():

    w, mesh = nonlinear_mixedfe(automatic_jacobian=True)

    verify_against_ghia1982(w, mesh)
    
    
def test_nonlinear_mixedfe_manual_jacobian():

    w, mesh = nonlinear_mixedfe(automatic_jacobian=False)
    
    verify_against_ghia1982(w, mesh)

    
if __name__=='__main__':
    
    test_nonlinear_mixedfe_automatic_jacobian()
    
    test_nonlinear_mixedfe_manual_jacobian()
    