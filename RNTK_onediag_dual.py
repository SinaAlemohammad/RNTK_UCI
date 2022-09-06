import numpy as np
import jax
import symjax
import symjax.tensor as T
import jax.numpy as jnp
import time

def create_func(dic, printbool = False):
    N = int(dic["n_patrons1="])
    ti_length = int(dic["n_entradasTi="])
    ti_prime_length = int(dic["n_entradasTiP="])
    DATA = T.Placeholder((N, ti_length), 'float32', name = "X")
    DATAPRIME = T.Placeholder((N, ti_prime_length), 'float32', name = "X")
    # x = DATA[:,0]
    # X = x*x[:, None]
    # n = X.shape[0]

    rntkod = RNTK(dic, DATA, DATAPRIME) #could be flipped 

    start = time.time()
    kernels_ema = rntkod.create_func_for_diag()
    diag_func = symjax.function(DATA, DATAPRIME, outputs=kernels_ema)
    if printbool:
        print("time to create symjax", time.time() - start)

    return diag_func, rntkod
    # return None, rntkod

create_T_list = lambda vals: T.concatenate([T.expand_dims(i, axis = 0) for i in vals])

class RNTK():
    def __init__(self, dic, DATA, DATAPRIME, simple = False):
        if not simple:
            self.dim_1 = dic["n_entradasTiP="]
            self.dim_2 = dic["n_entradasTi="]
            self.dim_num =self.dim_1 + self.dim_2 + 1
            self.DATA = DATA
            self.DATAPRIME = DATAPRIME
            self.N = int(dic["n_patrons1="])
        self.sw = 1.142 #sqrt 2 (1.4) - FIXED
        self.su = 0.5 #[0.1,0.5,1] - SEARCH
        self.sb = 0.1 #[0, 0.2, 0.5] - SEARCH
        self.sh = 1 #[0, 0.5, 1] - SEARCH
        self.sv = 1 #1 - FIXED
        self.L = 1 #1 - FIXED
        self.Lf = 0 #0 - FIXED
        if not simple:

            self.qt = self.compute_q(self.DATA)
            self.qtprime = self.compute_q(self.DATAPRIME)

            clip_num = min(self.dim_1, self.dim_2) + 1
            middle_list = np.zeros(self.dim_num-(2 * clip_num) + 1)
            middle_list.fill(clip_num)
            self.dim_lengths = np.concatenate([np.arange(1,clip_num), middle_list, np.arange(clip_num, 0, -1)])

            self.how_many_before = [sum(self.dim_lengths[:j]) for j in range(0, len(self.dim_lengths))]

            length_betw = (self.dim_lengths - 1)[1:-1]
            self.ends_of_calced_diags = np.array([sum(length_betw[:j]) for j in range(0, len(length_betw)+1)])[1:] - 1

    def alg1_VT_dep(self, M): #here i will use M as the previous little q ////// NxN, same value for every row
        A = T.diag(M)  # GP_old is in R^{n*n} having the output gp kernel
        # of all pairs of data in the data set
        B = A * A[:, None]
        C = T.sqrt(B)  # in R^{n*n}
        D = M / C  # this is lambda in ReLU analyrucal formula (c in alg)
        E = T.clip(D, -1, 1)  # clipping E between -1 and 1 for numerical stability.
        F = (1 / (2 * np.pi)) * (E * (np.pi - T.arccos(E)) + T.sqrt(1 - E ** 2)) * C
        G = (np.pi - T.arccos(E)) / (2 * np.pi)
        return F,G

    def alg1_VT(self, Q):
        F = Q/2;
        # G = 0.5*(Q/Q)
        return F

    def alg2_VT(self, Qx, Qxprime, M): #K1, K2, K3
        B = T.outer(Qx, Qxprime)
        C = T.sqrt(B)# in R^{n*n}
        D = M / C  # this is lamblda in ReLU analyrucal formula
        E = np.clip(D, -1, 1) # clipping E between -1 and 1 for numerical stability.
        F = (1 / (2 * np.pi)) * (E * (np.pi - T.arccos(E)) + T.sqrt(1 - E ** 2)) * C
        G = (np.pi - T.arccos(E)) / (2 * np.pi)
        return F,G

    def compute_q(self, DATA):
        DATAT = T.transpose(DATA)
        xz = DATAT[0]
        # print(xz)
        init = self.su * 2 * T.linalg.norm(T.expand_dims(xz, axis = 0), ord = 2, axis = 0) + self.sb**2 + self.sh**2
        # print("init", init)
        # init = self.su * 2 * T.linalg.norm(xz, ord = 2) + self.sb**2 + self.sh**2 #make this a vectorized function 

        def scan_func(prevq, MINIDATAT): #MINIDATAT shuold be a vector of lenght N
            # print(MINIDATAT)
            # the trick to this one is to use the original VT
            S = self.alg1_VT(prevq) # -> M is K3 
            # S = prevq
            # newq = self.su * 2 * T.linalg.norm(T.expand_dims(xz, axis = 0), ord = 2, axis = 0) + self.sb**2 + self.sh**2
            newq = self.sw*2 * S + self.su * 2 * T.linalg.norm(T.expand_dims(MINIDATAT, axis = 0), ord = 2, axis = 0) + self.sb**2
            # print("newq", newq)
            return newq, newq
        
        last_ema, all_ema = T.scan(scan_func, init = init, sequences = [DATAT[1:]])
        return T.concatenate([T.expand_dims(init, axis = 0), all_ema])
    
    def get_diag_indices(self, jnpbool = False, printbool = False):
        switch_flag = 1
        dim_1_i = self.dim_1
        dim_2_i = 0

        tiprimes = []
        tis = []

        if printbool:
            print(dim_1_i, dim_2_i, switch_flag)
        for d in range(0,self.dim_num):
            tiprime = dim_1_i
            ti = dim_2_i

            # diag_func(tiprime, ti, dim_1, dim_2, rntk)
            tiprimes.append(tiprime)
            tis.append(ti)

            if dim_1_i == 0:
                switch_flag -= 1
            else:
                dim_1_i = dim_1_i - 1
            if switch_flag <= 0:
                dim_2_i = dim_2_i + 1
        if jnpbool:
            return jnp.array(tiprimes), jnp.array(tis)    
        return np.array(tiprimes), np.array(tis)

    # generates array of non-bc values (length is the number of points in matrix - number of dimensions)
    
    def make_boundary_condition(self, X):
        bc = self.sh ** 2 * self.sw ** 2 * T.eye(self.N, self.N) + (self.su ** 2)* X + self.sb ** 2 ## took out X || 
        # single_boundary_condition = T.expand_dims(bc, axis = 0)
        # single_boundary_condition = T.expand_dims(T.Variable((bc), "float32", "boundary_condition"), axis = 0)
        # boundary_condition = T.concatenate([single_boundary_condition, single_boundary_condition])
        return bc

    def compute_kernels(self, qtph, qtprimeph, qtidx, qtprimeidx, cur_lambda, cur_phi, prev_K, prev_theta):
        S_kernel, D_kernel = self.alg2_VT(qtph[qtidx], qtprimeph[qtprimeidx], cur_lambda)
        ret_K = prev_K + self.sv**2 * S_kernel#get current lamnda, get current qtph and qtprimeph
        ret_theta = prev_theta + self.sv**2 * S_kernel + self.sv**2 * D_kernel * cur_phi 
        return ret_K, ret_theta
    
    def create_func_for_diag(self):

        NEW_DATA = self.reorganize_data()
        NEW_DATA_ATTACHED = jnp.array(list(zip(NEW_DATA[:-1], NEW_DATA[1:])))
        # print(NEW_DATA_ATTACHED)

        x = self.DATA[:,0]
        X = x*x[:, None]

        boundary_condition = self.make_boundary_condition(X)
        
        # #lets create the inital kernels - should be starting with top right
        temp_K, temp_theta = self.compute_kernels(self.qt, self.qtprime, 0, self.dim_1, boundary_condition, boundary_condition, T.empty((self.N, self.N)), T.empty((self.N, self.N)))
        init_K, init_theta = self.compute_kernels(self.qt, self.qtprime, 0, self.dim_1 - 1, boundary_condition, boundary_condition, temp_K, temp_theta)

        initial_conditions = create_T_list([boundary_condition, boundary_condition, init_K, init_theta])
        # initial_conditions = create_T_list([T.empty((self.N, self.N)), T.empty((self.N, self.N)), T.empty((self.N, self.N)), T.empty((self.N, self.N)) + 2])
        
        ## prev_vals - (4,self.N,self.N) - previous phi, lambda, and the two kernel values
        ## idx - where we are on the diagonal
        def fn(prev_vals, idx, data_idxs, DATAPH, DATAPRIMEPH, qtph, qtprimeph):

            xTP = DATAPRIMEPH[data_idxs[0][0]] #N1
            xT = DATAPH[data_idxs[0][1]] #N2
            xINNER = T.inner(xT, xTP) #N1 x N2

            prev_lambda = prev_vals[0]
            prev_phi = prev_vals[1]
            prev_K = prev_vals[2]
            prev_theta = prev_vals[3]
            ## not boundary condition
            # print(qtph[data_idxs[0][1] - 1])
            # print(qtprimeph[data_idxs[0][0] - 1])
            S, D = self.alg2_VT(qtph[data_idxs[0][1] - 1], qtprimeph[data_idxs[0][0] - 1] ,prev_lambda)
            new_lambda = self.sw ** 2 * S + self.su ** 2 * xINNER + self.sb ** 2 ## took out an X
            new_phi = new_lambda + self.sw ** 2 * prev_phi * D
            # new_phi = prev_phi
            # new_lambda = prev_lambda

            #compute kernels
            S_kernel, D_kernel = self.alg2_VT(qtph[data_idxs[0][1]], qtprimeph[data_idxs[0][0]], new_lambda)
            new_K = prev_K + self.sv**2 * S_kernel#get current lamnda, get current qtph and qtprimeph
            new_theta = prev_theta + self.sv**2 * S_kernel + self.sv**2 * D_kernel * new_phi #TODO
            # ret_K = prev_K
            # ret_theta = prev_theta

            equal_check = lambda e: T.equal(idx, e)

            equal_result = sum(np.vectorize(equal_check)(self.ends_of_calced_diags)) > 0

            def true_f(k,t, qp, gph, dataph, dataprimeph, di): 
                xTP_NEXT = dataprimeph[di[1][0]]
                xT_NEXT = dataph[di[1][1]]
                xINNER_NEXT = T.inner(xT_NEXT, xTP_NEXT)
                new_bc = self.make_boundary_condition(xINNER_NEXT)
                ret_lambda = ret_phi = new_bc 

                S_bc_kernel, D_bc_kernel = self.alg2_VT(qp[di[1][1]], gph[di[1][0]], ret_lambda)
                ret_K = k + self.sv**2 * S_bc_kernel#get current lamnda, get current qtph and qtprimeph
                ret_theta = t + self.sv**2 * S_bc_kernel + self.sv**2 * D_bc_kernel * ret_phi #TODO

                return ret_lambda, ret_phi, ret_K, ret_theta
            false_f = lambda l,p,k,t: (l,p,k,t)

            ret_lambda, ret_phi, ret_K, ret_theta = T.cond(equal_result, true_f, false_f, [new_K, new_theta, qtph, qtprimeph, DATAPH, DATAPRIMEPH, data_idxs], [new_lambda, new_phi, new_K, new_theta])
            
            to_carry = create_T_list([ret_lambda, ret_phi, ret_K, ret_theta])
            # print('got poast second create list')
            
            return to_carry, np.array(())
        
        carry_ema, _ = T.scan(
            fn, init =  initial_conditions, sequences=[jnp.arange(0, sum(self.dim_lengths) - self.dim_num), NEW_DATA_ATTACHED], non_sequences=[T.transpose(self.DATA), T.transpose(self.DATAPRIME), self.qt, self.qtprime]
        )
        
        return carry_ema[2:4] ## so here, the output will be the added up kernels except for the boundary conditions
        # return self.compute_kernels(all_ema)

    def reorganize_data(self, printbool = False):
        TiPrimes, Tis = self.get_diag_indices(printbool = printbool)
        reorganized_data = []
        for diag_idx in range(1, self.dim_num - 1):
            TiP = TiPrimes[diag_idx]
            Ti = Tis[diag_idx]

            dim_len = self.dim_lengths[diag_idx]
            for diag_pos in range(1, int(dim_len)):
                # we should never see 0 here, since those are reserved for boundary conditions
                if printbool:
                    print(f"taking position {TiP + diag_pos} from TiP, {Ti + diag_pos} from Ti")
                reorganized_data.append((TiP + diag_pos, Ti + diag_pos))
                # reorganized_data = self.add_or_create(reorganized_data, T.concatenate([T.expand_dims(self.DATA[:,TiP + diag_pos], axis = 0), T.expand_dims(self.DATA[:,Ti + diag_pos], axis = 0)]))
        reorganized_data.append((0,0))
        return jnp.array(reorganized_data)


    # def diag_func_wrapper(self, dim_1_idx, dim_2_idx, fbool = False, jmode = False):
    #     # print('tests')
    #     f = self.create_func_for_diag(dim_1_idx, dim_2_idx, function = fbool, jmode = jmode)
    #     # print('teste')
    #     if fbool:
    #         return f(np.arange(0,min(self.dim_1, self.dim_2) - (dim_1_idx + dim_2_idx)), self.dim_1, self.dim_2, dim_1_idx, dim_2_idx, self.n)
    #     return f

    # def add_or_create(self, tlist, titem):
    #     if tlist is None:
    #         return T.expand_dims(titem, axis = 0)
    #     else:
    #         return T.concatenate([tlist, T.expand_dims(titem, axis = 0)])

    # def get_ends_of_diags(self, result_ema):
    #     # ends_of_diags = None
    #     # for end in self.ends_of_calced_diags:
    #     #     index_test = result_ema[int(end)]
    #     #     ends_of_diags = self.add_or_create(ends_of_diags, index_test)
    #     ends_of_diags = result_ema[self.ends_of_calced_diags.astype('int')]
    #     prepended = T.concatenate([T.expand_dims(self.boundary_condition, axis = 0), ends_of_diags])
    #     return T.concatenate([prepended, T.expand_dims(self.boundary_condition, axis = 0)])


    # def compute_kernels(self, final_ema):

    #     diag_ends = self.get_ends_of_diags(final_ema)

    #     S_init, D_init = self.alg1_VT(diag_ends[0][0])
    #     init_Kappa  = self.sv ** 2 * S_init
    #     init_Theta = init_Kappa + self.sv ** 2 * diag_ends[0][1] * D_init
    #     init_list = T.concatenate([T.expand_dims(init_Kappa, axis = 0), T.expand_dims(init_Theta, axis = 0)])

    #     def map_test(gp_rntk_sum, gp_rntk):
    #         S, D = self.alg1_VT(gp_rntk[0])
    #         ret1 = self.sv ** 2 * S
    #         ret2 = ret1 + self.sv ** 2 * gp_rntk[1] * D
    #         gp_rntk_sum = T.index_add(gp_rntk_sum,0, ret1)
    #         gp_rntk_sum = T.index_add(gp_rntk_sum,1, ret2)

    #         return gp_rntk_sum, gp_rntk_sum

    #     final_K_T, inter_results = T.scan(
    #                 map_test, init =  init_list, sequences=[diag_ends[1:]]
    #             )
    #     self.test = final_K_T
    #     return final_K_T


    # def no_bc_arrays_to_diag(self, input_array):

        # indices_to_set = np.sort(list(set(range(0,int(sum(self.dim_lengths)))) - set(self.how_many_before)))
        # array_of_diags = np.zeros(int(sum(self.dim_lengths)), dtype = "object")
        # array_of_diags.fill(self.boundary_condition)
        # np.put(array_of_diags, indices_to_set, [input_array[i] for i in range(input_array.shape[0])])

        # full_lambda = None
        # full_phi = None
        # for i in range(0, self.dim_1 + 1): #these are rows
        #     column_lambda = None
        #     column_phi = None
        #     for j in range(0, self.dim_2 + 1): #these are columns
        #         list_index = min(self.dim_1-i, j) #could be dim 1
        #         which_list = j + i
        #         new_list_idx = list_index + int(self.how_many_before[which_list])
        #         column_lambda = self.add_or_create(column_lambda, array_of_diags[new_list_idx][0])
        #         column_phi = self.add_or_create(column_phi, array_of_diags[new_list_idx][1])
        #         # column_lambda.append()
        #         # column_phi.append(array_of_diags[new_list_idx][1])
        #     full_lambda = self.add_or_create(full_lambda, column_lambda)
        #     full_phi = self.add_or_create(full_phi, column_phi)
        #     # full_lambda.appe nd(column_lambda)
        #     # full_phi.append(column_phi)
        # return full_lambda, full_phi

    # def old_no_bc_arrays_to_diag(self, input_array):

        # indices_to_set = np.sort(list(set(range(0,int(sum(self.dim_lengths)))) - set(self.how_many_before)))
        # array_of_diags = np.zeros(int(sum(self.dim_lengths)), dtype = "object")
        # array_of_diags.fill(self.boundary_condition)
        # np.put(array_of_diags, indices_to_set, [input_array[i] for i in range(input_array.shape[0])])

        # full_lambda = []
        # full_phi = []
        # for i in range(0, self.dim_1 + 1): #these are rows
        #     column_lambda = []
        #     column_phi = []
        #     for j in range(0, self.dim_2 + 1): #these are columns
        #         list_index = min(self.dim_1-i, j) #could be dim 1
        #         which_list = j + i
        #         new_list_idx = list_index + int(self.how_many_before[which_list])
        #         column_lambda.append(array_of_diags[new_list_idx][0])
        #         column_phi.append(array_of_diags[new_list_idx][1])
        #     full_lambda.append(column_lambda)
        #     full_phi.append(column_phi)
        # return np.array(full_lambda), np.array(full_phi)