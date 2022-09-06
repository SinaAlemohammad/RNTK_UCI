import numpy as np
import jax
import symjax
import symjax.tensor as T

class RNTK():
    def __init__(self, N, length, param):
        self.sw = param['sigmaw']
        self.su = param['sigmau']
        self.sb = param['sigmab']
        self.sh = param['sigmah']
        self.L = param['L']
        self.Lf = param['Lf']
        self.sv = param['sigmav']
        self.N = N
        self.length = length

    def RNTK_function(self):
        print(f"N, {self.N}, length, {self.length}")
        DATA = T.Placeholder((self.N, self.length), 'float32')
        RNTK,GP = self.RNTK_first(DATA[:,0])
        v, _ = T.scan(lambda a,b:self.RNTK_middle(a,b),sequences=[ T.transpose(DATA[:, 1:]) ], init=T.stack([RNTK,GP]))
        RNTK_last,RNTK_avg = self.RNTK_output(v)
        f = symjax.function(DATA, outputs= [RNTK_last,RNTK_avg])
        return RNTK_last,RNTK_avg

    def RNTK_first(self,x): # alg 1, line 1
        X = x*x[:, None]
        n = X.shape[0] #       // creates a diagonal matrix of sh^2 * sw^2
        test = self.sh ** 2 * self.sw ** 2 * T.eye(n, n) + (self.su ** 2) * X + self.sb ** 2
        gp_new = T.expand_dims(test, axis = 0) # line 2, alg 1 #GP IS GAMMA, RNTK IS PHI
        rntk_new = gp_new
        print("gp_new 1", gp_new)
        print("rntk_new 1", rntk_new)
        for l in range(self.L-1): #line 3, alg 1
            l = l+1
            print("gp_new", gp_new[l-1])
            S_new,D_new = self.VT(gp_new[l-1]) 
            gp_new = T.concatenate([gp_new,T.expand_dims(self.sh ** 2 * self.sw ** 2 * T.eye(n, n) + self.su**2 * S_new + self.sb**2,axis = 0)]) #line 4, alg 1
            rntk_new = T.concatenate([rntk_new,T.expand_dims(gp_new[l] + (self.Lf <= (l-1))*self.su**2*rntk_new[l-1]*D_new,axis = 0)])
        S_old,D_old = self.VT(gp_new[self.L-1])
        gp_new = T.concatenate([gp_new,T.expand_dims(self.sv**2*S_old,axis = 0)]) #line 5, alg 1
        rntk_new = T.concatenate([rntk_new,T.expand_dims(gp_new[self.L] + (self.Lf != self.L)*self.sv**2*rntk_new[self.L-1]*D_old,axis = 0)])
        print("gp_new 2", gp_new)
        print("rntk_new 2", rntk_new)
        return rntk_new, gp_new

    def RNTK_middle(self, previous,x): # line 7, alg 1
        X = x * x[:, None] # <x, x^t>
        rntk_old = previous[0]
        gp_old = previous[1]
        S_old,D_old = self.VT(gp_old[0])#.  //vv K(1,t-1)
        gp_new = T.expand_dims(self.sw ** 2 * S_old + (self.su ** 2) * X + self.sb ** 2,axis = 0) # line 8, alg 1
        if self.Lf == 0: # if none of the katers are fixed, use the standard
            rntk_new = T.expand_dims(gp_new[0] + self.sw**2*rntk_old[0]*D_old,axis = 0)
        else:
            rntk_new = T.expand_dims(gp_new[0],axis = 0) 
        
        print("gp_new 3", gp_new)
        print("rntk_new 3", rntk_new)

        for l in range(self.L-1): #line 10
            l = l+1
            S_new,D_new = self.VT(gp_new[l-1]) # l-1, t
            S_old,D_old = self.VT(gp_old[l]) # t-1, l
            gp_new = T.concatenate( [gp_new, T.expand_dims( self.sw ** 2 * S_old + self.su ** 2 * S_new +  self.sb ** 2, axis = 0)]) #line 10
            rntk_new = T.concatenate( [ rntk_new,  T.expand_dims( gp_new[l] +(self.Lf <= l)*self.sw**2*rntk_old[l]*D_old +(self.Lf <= (l-1))* self.su**2*rntk_new[l-1]*D_new  ,axis = 0)   ]  )
        S_old,D_old = self.VT(gp_new[self.L-1])
        gp_new = T.concatenate([gp_new,T.expand_dims(self.sv**2*S_old,axis = 0)]) # line 11
        rntk_new = T.concatenate([rntk_new,T.expand_dims(rntk_old[self.L]+ gp_new[self.L] + (self.Lf != self.L)*self.sv**2*rntk_new[self.L-1]*D_old,axis = 0)])
        print("gp_new 4", gp_new)
        print("rntk_new 4", rntk_new)
        return T.stack([rntk_new,gp_new]),x

    def RNTK_output(self, previous):
        rntk_old = previous[0]
        gp_old = previous[1]
        S_old,D_old = self.VT(gp_old[self.L-1])
        RNTK_last  = self.sv**2*S_old + (self.Lf != self.L)*self.sv**2*rntk_old[self.L-1]*D_old 
        RNTK_avg =  (RNTK_last + rntk_old[self.L])/self.length
        return RNTK_last,RNTK_avg

    def VT(self, M):
        A = T.diag(M)  # GP_old is in R^{n*n} having the output gp kernel
        # of all pairs of data in the data set
        B = A * A[:, None]
        C = T.sqrt(B)  # in R^{n*n}
        D = M / C  # this is lamblda in ReLU analyrucal formula
        E = T.clip(D, -1, 1)  # clipping E between -1 and 1 for numerical stability.
        F = (1 / (2 * np.pi)) * (E * (np.pi - T.arccos(E)) + T.sqrt(1 - E ** 2)) * C
        G = (np.pi - T.arccos(E)) / (2 * np.pi)
        return F,G
