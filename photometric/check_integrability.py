import numpy as np

def check_integrability(normals):
    #  CHECK_INTEGRABILITY check the surface gradient is acceptable
    #   normals: normal image
    #   p : df / dx
    #   q : df / dy
    #   SE : Squared Errors of the 2 second derivatives

    # initalization
    p = np.zeros(normals.shape[:2])
    q = np.zeros(normals.shape[:2])
    SE = np.zeros(normals.shape[:2])
    
    """
    ================
    Your code here
    ================
    Compute p and q, where
    p measures value of df / dx
    q measures value of df / dy
    
    """
    
    p = normals[:,:,0]/normals[:,:,2]
    q = normals[:,:,1]/normals[:,:,2]

    # change nan to 0
    p[p!=p] = 0
    q[q!=q] = 0
    
    """
    ================
    Your code here
    ================
    approximate second derivate by neighbor difference
    and compute the Squared Errors SE of the 2 second derivatives SE
    
    """
    dp = np.concatenate([
        np.diff(p, axis=1),
        np.zeros((p.shape[0], 1))
    ], axis=1)
    dq = np.concatenate([
        np.diff(q, axis=0),
        np.zeros((1, q.shape[1]))
    ], axis=0)
    
    SE = (dp - dq)**2


    return p, q, SE


if __name__ == '__main__':
    normals = np.zeros([10,10,3])
    check_integrability(normals)