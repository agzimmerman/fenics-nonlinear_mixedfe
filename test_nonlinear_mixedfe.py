""" This module tests nonlinear_mixefe.py """
import fenics
import nonlinear_mixedfe


def verify_against_ghia1982(w):
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

    data = {'Re': 100, 'x': 0.5,
        'y': [1.0000, 0.9766, 0.1016, 0.0547, 0.0000],
        'ux': [1.0000, 0.8412, -0.0643, -0.0372, 0.0000]}
    
    for i, true_ux in enumerate(data['ux']):
    
        p = fenics.Point(data['x'], data['y'][i])
    
        wval = w.leaf_node()(p)
        
        ux = wval[0]
        
        absolute_error = ux - true_ux
        
        if abs(true_ux) > fenics.DOLFIN_EPS:
        
            relative_error = abs(absolute_error)/abs(true_ux)

            assert(relative_error < 0.01)
            
        else:
        
            assert(absolute_error < fenics.DOLFIN_EPS)

    print("Verified successfully against Ghia1982.")
    
    
def test_nonlinear_mixedfe_automatic_jacobian():
    """ Test the solver with an automatic Jacobian."""
    w = nonlinear_mixedfe.nonlinear_mixedfe(automatic_jacobian=True)

    verify_against_ghia1982(w)


def test_nonlinear_mixedfe_manual_jacobian():
    """ Test the solver with the manually implemented Jacobian."""
    w = nonlinear_mixedfe.nonlinear_mixedfe(automatic_jacobian=False)
    
    verify_against_ghia1982(w)
    
    
if __name__=='__main__':
    
    test_nonlinear_mixedfe_automatic_jacobian()
    
    test_nonlinear_mixedfe_manual_jacobian()
