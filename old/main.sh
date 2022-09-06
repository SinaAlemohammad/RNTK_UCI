path_to_log=log_hyperparam
gpu_id=0
sigmaw=(1.4142)
sigmau=(0.25 0.5)
sigmab=(0.001 0.1)
sigmah=(0.0) #0, -> 0.3 // should get 16 different kernels, get bidirectional for 32
for dataset_id in {0..0}  #$dataset_ids # used to be 89
do
  for sw in ${sigmaw[@]}
  do
    for su in ${sigmau[@]}
    do
      for sb in ${sigmab[@]}
      do
        for sh in ${sigmah[@]}
        do
          for l in {1..2}
          do
            python RNTK_UCI_hyperparam.py --dataset_id $dataset_id --sw $sw --su $su --sb $sb --sh $sh --L $l --gpu_id $gpu_id --path_to_log $path_to_log  --c 0.01 1 100 10000 1000000 --avg 0 1 --Lf 0 $l --Flip 0 1 2
          done
        done
      done
    done
  done
done

python best_hyperparam_vote.py --sigmaw 1.4142 --sigmau 0.25 0.5 --sigmab 0.001 0.25 --sigmah 0.0 --c 0.01 1 100 10000 1000000  --path_to_log $path_to_log  --L 2 --average 0 --Flip 0 1

for dataset_id in {0..0}  #$dataset_ids
do
   python RNTK_UCI_test_vote.py --dataset_id $dataset_id --gpu_id $gpu_id --path_to_log $path_to_log
done


python print_final_vote.py 
