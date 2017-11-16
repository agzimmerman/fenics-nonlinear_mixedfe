""" This is an example of how to use mixed finite elements to solve nonlinear problems """
import fenics


def nonlinear_mixedfe(automatic_jacobian=True, Re=100., m = 2,
        adaptive_solver_tolerance = 1.e-5):
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

    # Set physical parameters
    mu = 1./Re
    
    
    # Set numerical parameters.
    mesh = fenics.UnitSquareMesh(m, m, 'crossed')
    
    """ Parameter for pressure penalty formulation, should be on the order of 1.e-7-1.e-8"""
    gamma = 1.e-7  
    
    pressure_degree = 1


    # Set function spaces for the variational form
    """ Higher order velocity element needed for stability (Donea, Huerta 2003)"""
    velocity_degree = pressure_degree + 1

    velocity_element = fenics.VectorElement('P', mesh.ufl_cell(), velocity_degree)

    pressure_element = fenics.FiniteElement('P', mesh.ufl_cell(), pressure_degree)

    W_ele = fenics.MixedElement([velocity_element, pressure_element])

    W = fenics.FunctionSpace(mesh, W_ele)  
    
    
    # Set Dirichlet boundary conditions.
    bcs = [
        fenics.DirichletBC(W.sub(0), (1., 0.), 
            'near(x[1],  1.)'),
        fenics.DirichletBC(W.sub(0), (0., 0.),
            'near(x[0],  0.) | near(x[0],  1.) | near(x[1],  0.)'),
        fenics.DirichletBC(W.sub(1), 0.,
            'near(x[0], 0.) && near(x[1], 0.)',
            method='pointwise')]    
    
    
    # Set nonlinear variational form.
    inner, dot, grad, div, sym = fenics.inner, fenics.dot, fenics.grad, fenics.div, fenics.sym
    
    D = lambda u : sym(grad(u))
    
    a = lambda u, v : 2.*mu*inner(D(u), D(v))
    
    b = lambda u, q : -div(u)*q
    
    c = lambda w, z, v : dot(dot(grad(z), w), v)
    
    
    # Write the nonlinear variational form.
    v, q = fenics.TestFunctions(W)
        
    w = fenics.Function(W)
    
    u, p = fenics.split(w)
    
    F = (
        b(u, q) - gamma*p*q
        + c(u, u, v) + a(u, v) + b(v, p)
        )*fenics.dx
    
    
    # Write the Jacobian in variational form.
    dw = fenics.TrialFunction(W)  # Residual, solution of the Newton linearized system
    
    if automatic_jacobian:
    
        JF = fenics.derivative(F, w, dw)
        
    else: 
        """Manually implement the exact Gateaux derivative.
        This is both more robust and more computationally efficient,
        at the cost of having to do some basic calculus."""
        du, dp = fenics.split(dw)
          
        JF = (
            b(du, q) - gamma*dp*q
            + c(u, du, v) + c(du, u, v) + a(du, v) + b(v, dp)
            )*fenics.dx
    
    
    # Write the error measure for adaptive mesh refinement.
    M = u[0]*u[0]*fenics.dx
    
    
    # Solve nonlinear problem.
    problem = fenics.NonlinearVariationalProblem(F, w, bcs, JF)

    solver  = fenics.AdaptiveNonlinearVariationalSolver(problem, M)

    solver.solve(adaptive_solver_tolerance)
    
    
    # Write the solution to disk for visualization.
    solution_files = [fenics.File('velocity.pvd'), fenics.File('pressure.pvd')]
    
    velocity, pressure = w.split()
    
    velocity.rename("u", "velocity")

    pressure.rename("p", "pressure")

    for i, var in enumerate([velocity, pressure]):

        solution_files[i] << var

    
    # Return the solution and the mesh for verification or other purposes.
    return w

    
if __name__=='__main__':
    
    nonlinear_mixedfe()

    