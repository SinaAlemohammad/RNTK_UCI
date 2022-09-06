import os
import json
import sys
import numpy as np
import jax
import symjax
import symjax.tensor as T
import argparse
import RNTK_avg
import tools
import sklearn
from sklearn.svm import SVC
from copy import deepcopy
from scipy import stats

parser = argparse.ArgumentParser(description='user input to the rntk experiment')
parser.add_argument('--dataset_id', type=int, default=0, help='dataset index')
parser.add_argument('--gpu_id', type=float, default=1, help='which gpu to run experiment')
parser.add_argument('--path_to_log', type=str, default='log_hyperparam', help='dataset path') # XXXXX could be log_hyperparam or log_final
args = parser.parse_args()
dataset_id = args.dataset_id
path_to_log = args.path_to_log
gpu_id = args.gpu_id
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

def printProgressBar(i,max,postText):
    n_bar = 100 #size of progress bar
    j= i/max
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(n_bar * j):{n_bar}s}] {int(100 * j)}%  {postText}")
    sys.stdout.flush()


datasets = []
for i in map(lambda x : x.split(), open("data/datasets.txt", "r").readlines()):
    datasets.append(i[0])
dataset = datasets[dataset_id]
dic = dict()
for k, v in map(lambda x : x.split(), open("data" + "/" + dataset + "/" + dataset + ".txt", "r").readlines()):
    dic[k] = v
n_class = int(dic["n_clases="])
d = int(dic["n_entradas="])
n_train = int(dic["n_patrons_entrena="])
n_test = int(dic["n_patrons_valida="])
n_total = int(dic["n_patrons1="])


f = open("data/" + dataset + "/" + dic["fich1="], "r").readlines()[1:]
X = np.asarray(list(map(lambda x: list(map(float, x.split()[1:-1])), f)))
X = tools.normalizeData(X)
y = np.asarray(list(map(lambda x: int(x.split()[-1]), f)))


with open(os.path.join(path_to_log, 'imax-dataset_{}'.format(dataset))) as f:
     result = json.load(f)
i_max = result['i_max']
if (i_max % 2) == 0:
   i_max = i_max-1

fold = list(map(lambda x: list(map(int, x.split())), open("data/" + dataset + "/" + "conxuntos_kfold.dat", "r").readlines()))
Z = [None]*4
Y = [None]*4
for repeat in range(4):
    train_fold, test_fold = fold[repeat * 2], fold[repeat * 2 + 1]
    Z[repeat] = np.zeros((i_max,len(test_fold)))
    Y[repeat] = y[test_fold]



for i in range(1,i_max+1):
   with open(os.path.join(path_to_log, 'UCI-rntk-val-rntk-dataset_{}-i_{}'.format(dataset,i))) as f:
        result = json.load(f)

   flip = result['flip']
   param = {}
   param['sigmaw'] = result['sw']
   param['sigmau'] = result['su']
   param['sigmab'] = result['sb']
   param['sigmah'] = result['sh']
   param['sigmav'] = 1
   param['L'] = result['L']
   param['Lf'] = result['Lf']
   cost = result['c']
   avg = result['avg']

   f = RNTK_avg.RNTK_function(n_total,d,param)
   if flip == 2:
     Kb0 = np.array(f(tools.Augdata(X,0)),dtype = object)
     Kb1 = np.array(f(tools.Augdata(X,1)),dtype = object)
     Kb = [None]*2
     Kb[0]=Kb0[0] + Kb1[0]
     Kb[1]=Kb0[1] + Kb1[1]
   else:
     Kb = np.array(f(tools.Augdata(X,flip)),dtype = object) 
   K = Kb[avg]
   for repeat in range(4):
       train_fold, test_fold = fold[repeat * 2], fold[repeat * 2 + 1]
       Z[repeat][i-1,:] = tools.svm2(K[train_fold][:, train_fold], K[test_fold][:, train_fold], y[train_fold], cost, n_class) 
   printProgressBar(i,i_max,dataset)
z = [None]*4
acc = 0
for repeat in range(4):
    z[repeat] = stats.mode(Z[repeat])[0][0]
    acc += 0.25*100*np.sum(z[repeat] == Y[repeat])/len(Y[repeat])

test_rntk = {
    'dataset': dataset,
    'n': n_total,
    'd': d,
    'n_class': n_class,
    'acc_val': result['acc'],
    'acc_test': acc
}
with open(os.path.join(path_to_log, 'UCI-vote-rntk-final-rntk-dataset_{}'.format(
                                            dataset)), 'w') as f:
    f.write(json.dumps(test_rntk))
    
 
print(':',' acc:', np.round(acc,2),'  ***************************')

