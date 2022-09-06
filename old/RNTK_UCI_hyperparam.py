import os
import json
import sys
import pickle as pkl
import numpy as np
import jax
import symjax
import symjax.tensor as T
import sklearn
from sklearn.svm import SVC
import argparse
from copy import deepcopy
import RNTK_avg
import tools

parser = argparse.ArgumentParser(description='user input to the rntk experiment')
parser.add_argument('--dataset_id', type=int, default=0, help='dataset index')
parser.add_argument('--sw', type=float, default=1.41, help='sigmaw')
parser.add_argument('--su', type=float, default=0.1, help='sigmau')
parser.add_argument('--sb', type=float, default=0, help='sigmab')
parser.add_argument('--sh', type=float, default=0, help='sigmah')
parser.add_argument('--L', type=int, default=1, help='maximum number of layers')
parser.add_argument('--gpu_id', type=float, default= 1, help='which gpu to run experiment')
parser.add_argument('--path_to_log', type=str, default='XXXXX', help='path to save the sclassifiers informations') 
parser.add_argument('--c', nargs='+',type=float,help='svm cost') 
parser.add_argument('--Flip', nargs='+',type=int,help='fliping the data: 0 flips the data and 1 doesnt make any changes, and 2 is the bi directional form ') 
parser.add_argument('--avg', nargs='+',type=int,help='0 the output is the last hidden state, 1 is the average of all hidden states')
parser.add_argument('--Lf', nargs='+',type=int,help='Fixes the first Lf layers in training, 0 gives rntk and L gives the GP kernel')
args = parser.parse_args()
dataset_id = args.dataset_id
path_to_log = args.path_to_log
sw = args.sw
su = args.su
sb = args.sb
sh = args.sh
L = args.L
gpu_id = args.gpu_id
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
c = args.c
Flip = args.Flip
Average = args.avg
Lf = args.Lf


datasets = [i[0] for i in map(lambda x : x.split(), open("data/datasets.txt", "r").readlines())]
dataset = datasets[dataset_id]


dic = {k:v for k,v in map(lambda x : x.split(), open("data" + "/" + dataset + "/" + dataset + ".txt", "r").readlines())}

n_class = int(dic["n_clases="])
d = int(dic["n_entradas="])
n_train = int(dic["n_patrons_entrena="])
n_test = int(dic["n_patrons_valida="])
n_total = int(dic["n_patrons1="])

f = open("data/" + dataset + "/" + dic["fich1="], "r").readlines()[1:]
X = np.asarray(list(map(lambda x: list(map(float, x.split()[1:-1])), f)))
X = tools.normalizeData(X)
y = np.asarray(list(map(lambda x: int(x.split()[-1]), f)))
fold = list(map(lambda x: list(map(int, x.split())), open("data" + "/" + dataset + "/" + "conxuntos.dat", "r").readlines()))
train_fold, val_fold = fold[0], fold[1]


print (dataset, "\tN:", n_total, "\td:", d, "\tc:", n_class)



param = {}
param['sigmaw'] = sw
param['sigmau'] = su
param['sigmab'] = sb
param['sigmah'] = sh
param['sigmav'] = 1
param['L'] = L

for lf in Lf:
    param['Lf'] = lf
    print ('*********************','Lf:',param['Lf'],'L',L)
    f = RNTK_avg.RNTK(n_total,d,param).RNTK_function()
    for flip in Flip:
       if flip == 2:
          Kb0 = np.array(f(tools.Augdata(X,0)),dtype = object)
          Kb1 = np.array(f(tools.Augdata(X,1)),dtype = object)
          Kb = [None]*2
          Kb[0]=Kb0[0] + Kb1[0]
          Kb[1]=Kb0[1] + Kb1[1]
       else:
          Kb = np.array(f(tools.Augdata(X,flip)),dtype = object) 
       for avg in Average:
         K = Kb[avg]
         for cost in c:
            acc = 0
            if np.linalg.norm(K)/(n_total*n_total) < 10**12:
               acc = tools.svm(K[train_fold][:, train_fold], K[val_fold][:, train_fold], y[train_fold], y[val_fold], cost, n_class)
            result = {
             'dataset': dataset,
             'sw': sw,
             'su': su,
             'sb': sb,
             'sh': sh,
             'c' : cost, #special
             'L': L,
             'Lf':lf,
             'flip': flip,
             'acc': acc, #special
             'avg':avg
                }
            # print ('RNTK*********** data set:', dataset, ' sw:', sw,' su:',su,' sb:',sb,' sh:',sh,'L', L,'Lf', lf,'flip:',flip,'average:',avg ,'cost:', np.log10(cost), 'acc', np.round(100*acc,2), '************')
            with open(os.path.join(path_to_log,'UCI-rntk-dataset_{}-sw_{}-su_{}-sb_{}-sh_{}-L_{}-Lf_{}-flip_{}-avg_{}-c_{}'.format(
                    dataset, sw, su, sb, sh, L, lf,flip,avg,cost)),'w') as file:
                file.write(json.dumps(result))