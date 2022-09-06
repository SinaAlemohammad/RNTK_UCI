import os
import json
import numpy as np
path_to_log = 'log_hyperparam'

datasets = []
idx = 0
for i in map(lambda x : x.split(), open("data/datasets.txt", "r").readlines()):
    datasets.append(i[0])
acc_list = []
for dataset in datasets:
    with open(os.path.join(path_to_log, 'UCI-vote-rntk-final-rntk-dataset_{}'.format(dataset))) as f:
         result = json.load(f)
    acc = result['acc_test']
    print('dataset:',dataset,' test acc:', np.round(acc,10))
    acc_list.append(acc)
average_acc = np.mean(acc_list)
print(round(average_acc,2))
std_acc = np.std(acc_list)
print(round(std_acc,2))
