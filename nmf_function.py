import numpy as np
np.random.seed(1515)

def process(rank, V, W0 = None, lam=0, H0 = None,verbose=False, iterations=100):
    eps = np.spacing(1)
    W = W0 if W0 is not None else np.random.rand(V.shape[0],rank)+eps
    H = H0 if H0 is not None else np.random.rand(rank, V.shape[1])+eps
    for i in range(iterations):
        [V, W, H] = update(V, W, H, lam)
        err = compute_error(V, W, H)
        if verbose == True:
            print err
    return [W, H, err]
        
   
def compute_error(V, W, H):
    eps = np.spacing(1)
    R=np.dot(W,H)
    err = np.sum(np.multiply(V,np.log((V+eps)/(R+eps))) - V + R)
    return err        

def update(V, W, H, lam):
    eps = np.spacing(1)
    # if self.update_W:
        # R = np.dot(W,H)
        # W *= np.dot(np.divide(V, R + eps) , H.T) / (np.dot(self.ones, H.T) + eps)
    R = np.dot(W,H)
    H *= np.dot(W.T, np.divide(V, R + eps)) / (np.dot(W.T, np.ones(V.shape)) + lam + eps)
    return [V, W, H]
