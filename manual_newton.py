import fenics


def solve(A, L, w_k, bcs, max_newton_iterations=12):
            
    print("\nIterating Newton method")
    
    converged = False
    
    iteration_count = 0
    
    w_w = w_k.copy()
    
    for k in range(max_newton_iterations):

        fenics.solve(A == L, w_w, bcs)

        w_k.assign(w_k - w_w)
        
        norm_residual = fenics.norm(w_w, 'L2')/fenics.norm(w_k, 'L2')

        print("\nL2 norm of relative residual, || w_w || / || w_k || = "+str(norm_residual)+"\n")
        
        if norm_residual < newton_relative_tolerance:
            
            iteration_count = k + 1
            
            print("Converged after "+str(k)+" iterations")
            
            converged = True
            
            break
             
    return converged