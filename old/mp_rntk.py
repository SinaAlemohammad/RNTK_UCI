

def diag_func(tiprime_iter, ti_iter, dim_1_max, dim_2_max, rntk):
    while ((tiprime_iter <= dim_1_max) & (ti_iter <= dim_2_max)):
        print("inner iteration - ", tiprime_iter, ti_iter) #// DIM 1 IS T PRIME, DIM 2 IS T
        if ((ti_iter > 0) & (tiprime_iter > 0)):
            S, D = rntk.VT(lambdamatrix[0, ti_iter-1, tiprime_iter-1])
            lambdamatrix[0, ti_iter, tiprime_iter] = rntk.sw ** 2 * S + rntk.su ** 2 * X + rntk.sb ** 2
            phimatrix[0, ti_iter, tiprime_iter] = lambdamatrix[0, ti_iter, tiprime_iter] + rntk.sw ** 2 * phimatrix[0, ti_iter - 1, tiprime_iter - 1] * D
        else:
            test = rntk.sh ** 2 * rntk.sw ** 2 * T.eye(n, n) + (rntk.su ** 2) * X + rntk.sb ** 2
            # phimatrix[0,ti,tiprime_iter] = lambdamatrix[0,ti,tiprime_iter] = T.expand_dims(test, axis = 0) # line 2, alg 1
            phimatrix[0,ti_iter,tiprime_iter] = lambdamatrix[0,ti_iter,tiprime_iter] = test # line 2, alg 1

        tiprime_iter+=1
        ti_iter+=1