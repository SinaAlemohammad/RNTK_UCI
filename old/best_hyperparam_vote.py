import numpy as np
import os
import json
import sys
import argparse
from copy import deepcopy
import RNTK
import tools

parser = argparse.ArgumentParser(description='user input to the rntk experiment')
parser.add_argument('--sigmaw', nargs='+',type=float, default=1.4142, help='sigmaw')
parser.add_argument('--sigmau', nargs='+',type=float, default=0.25, help='sigmau')
parser.add_argument('--sigmah', nargs='+',type=float, default=0.0, help='sigmah')
parser.add_argument('--sigmab', nargs='+',type=float, default=0, help='sigmab')
parser.add_argument('--L', type=int, default=2, help='maximum number of layers')
parser.add_argument('--c', nargs='+',type=float, default=2, help='svm cost')  
parser.add_argument('--path_to_log', type=str, default='XXXXX', help='dataset path') 
parser.add_argument('--average', nargs='+',type=int, default=0, help='average pooling') 
parser.add_argument('--Flip', nargs='+',type=int, default=0, help='average pooling') 

args = parser.parse_args()
sigmaw = args.sigmaw
sigmau = args.sigmau
sigmab = args.sigmab
sigmah = args.sigmah
average = args.average
L = args.L
c = args.c
path_to_log = args.path_to_log
Flip = args.Flip


datasets = []
idx = 0
for i in map(lambda x : x.split(), open("data/datasets.txt", "r").readlines()):
      datasets.append(i[0])
      idx = idx+1
      
for dataset in datasets:
    acc_best = 0
    for avg in average:
       for flip in Flip:
          for sw in sigmaw:
             for sb in sigmab:
                for su in sigmau:
                   for sh in sigmah:
                      for l in range(L):
                          l = l+1
                          for lf in [l]:
                              for cost in c:
                                  with open(os.path.join(path_to_log,'UCI-rntk-dataset_{}-sw_{}-su_{}-sb_{}-sh_{}-L_{}-Lf_{}-flip_{}-avg_{}-c_{}'.format(
                                             dataset, sw, su, sb, sh,l,lf,flip,avg,cost))) as f:
                                       result = json.load(f)
                                  acc = result['acc']
                                  if acc > acc_best:
                                     acc_best = acc
                                     sw_best = sw
                                     su_best = su
                                     sb_best = sb
                                     sh_best = sh
                                     cost_best = cost
                                     l_best = l
                                     lf_best = lf
                                     flip_best = flip
                                     avg_best = avg
    i = 0
    for avg in average:
       for flip in Flip:
          for sw in sigmaw:
             for sb in sigmab:
                for su in sigmau:
                   for sh in sigmah:
                      for l in range(L):
                          l = l+1
                          for lf in [l]:
                              for cost in c:
                                  with open(os.path.join(path_to_log,'UCI-rntk-dataset_{}-sw_{}-su_{}-sb_{}-sh_{}-L_{}-Lf_{}-flip_{}-avg_{}-c_{}'.format(
                                             dataset, sw, su, sb, sh,l,lf,flip,avg,cost))) as f:
                                       result = json.load(f)
                                  acc = result['acc']
                                  if acc == acc_best:
                                     i = i + 1
                                     print('rntk***********dataset:', dataset,' best vall acc:', np.round(100*acc,2) ,' sw:',sw,' su:',su,' sb:',sb,' sh:',sh,'L',l,' Lf:',lf,' flip:',flip,'avg:',avg, 'cost:',cost,'i:',i,'*************')      
                                     mfval_rntk = {'dataset': dataset,'sw': sw,'su': su,'sb': sb,'sh': sh,'c' : cost,'L': l,'Lf': lf,'flip': flip,'avg': avg,'acc': acc, 'i':i}
                                     with open(os.path.join(path_to_log, 'UCI-rntk-val-rntk-dataset_{}-i_{}'.format(dataset,i)), 'w') as f:
                                          f.write(json.dumps(mfval_rntk))
    i_dataset = {'dataset': dataset, 'i_max':i }
    with open(os.path.join(path_to_log, 'imax-dataset_{}'.format(dataset)), 'w') as f:
          f.write(json.dumps(i_dataset))

