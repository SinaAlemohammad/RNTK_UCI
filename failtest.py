# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# from IPython import get_ipython

# %%
# get_ipython().run_line_magic('load_ext', 'line_profiler')
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# %%
import numpy as np
import jax.numpy as jnp
import old.tools, old.RNTK_avg
import jax
import symjax
import symjax.tensor as T
import copy
import time

from RNTK_onediag_dual import RNTK as RNTKOD_dual
from RNTK_onediag_dual import create_func as create_func_dual
from RNTK_onediag_stripped import RNTK as RNTKOD_stripped
from RNTK_onediag_stripped import create_func as create_func_stripped

import sys

# %%
# dataset = "trains"
# dataset = "bank"


# %%
# dic = {k:v for k,v in map(lambda x : x.split(), open("data" + "/" + dataset + "/" + dataset + ".txt", "r").readlines())}
dic = {}
dic["n_patrons1="] = 1#00 #// N (what affect symjax time)
dic["n_entradasTiP="] = 100 #99 works, 100 does not
dic["n_entradasTi="] = 100


# %%
def run_and_test(dic, type = "dual"):
    if not type == "dual":
        diag_func, rntk_dual = create_func_stripped(dic, True)
    else:
        diag_func, rntk_dual = create_func_dual(dic, True)
    fake_data_ti = np.zeros((dic["n_patrons1="],dic["n_entradasTi="]))
    fake_data_tiprime = np.zeros((dic["n_patrons1="],dic["n_entradasTiP="]))
    start = time.time()
    result = diag_func(fake_data_ti, fake_data_tiprime)
    # # gp_kernel = result[0]
    # # rntk_kernel = result[1]
    print("time to compute ", time.time() - start)
    return result, rntk_dual


result, rntk_dual = run_and_test(dic, "dual")
print(result)

# %%
# import json


# # %%
# # results = {}

# ti = int(sys.argv[1]) * 10
# tip = int(sys.argv[2]) * 10


# dic["n_patrons1="] = 20#00 #// N (what affect symjax time)
# dic["n_entradasTiP="] = tip#00 #// TiPrime length
# dic["n_entradasTi="] = ti#00 #// Ti length (multiplier on symjax time)
# #UCR dataset of about this size
# result = run_and_test(dic)
# array_sum = np.sum(result)
# array_has_nan = np.isnan(array_sum)
# # print(result.shape)
# with open("RNTK_UCI/timings20.txt", "a") as file:
#     json.dump({f"{ti},{tip}": int(array_has_nan)}, file, indent=1)
# with open("RNTK_UCI/results20.txt", "a") as file:
#     json.dump({f"{ti},{tip}": result.tolist()}, file, indent=1)
    # results[] = result


# %%
# diag_func, rntk_dual = create_func_dual(dic, True)

# %% [markdown]
# lets execute the symjax

# %%
# fake_data_ti = np.zeros((dic["n_patrons1="],dic["n_entradasTi="]))
# fake_data_tiprime = np.zeros((dic["n_patrons1="],dic["n_entradasTiP="]))


# %%
# start = time.time()
# result = diag_func(fake_data_ti, fake_data_tiprime)
# # # gp_kernel = result[0]
# # # rntk_kernel = result[1]
# print(time.time() - start)
# result


# %%
# %%timeit
# diag_func(fake_data)


# %%



