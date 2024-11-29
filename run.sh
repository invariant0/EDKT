

for random_seed in 0 1 2 3 4
do 
for group_id in 1 2 3 4
do

    python main.py --dataset pQSAR --group_id $group_id --encode_method GraphSAGE --random_seed $random_seed
    python main.py --dataset pQSAR --group_id $group_id --encode_method GraphGCN --random_seed $random_seed

done
done 


# for random_seed in 0 1 2 3 4
# do 
# python main.py --dataset fsmol --encode_method GraphGCN --num_encoder 2 --random_seed $random_seed
# done

