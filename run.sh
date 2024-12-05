# for random_seed in 0 1 2 3 4
# do 

for group_id in 1 2 3 4
do

    python main.py --dataset pQSAR --group_id $group_id --encode_method FP --num_encoder 50 --random_seed 0
    python main.py --dataset pQSAR --group_id $group_id --encode_method FPRGB --num_encoder 50 --random_seed 0
    # python main.py --dataset pQSAR --group_id $group_id --encode_method GraphGCN --random_seed $random_seed

done 

# done 

# for random_seed in {1..10}; do
#     python main.py --dataset fsmol --encode_method FPRGB --num_encoder 50 --random_seed $random_seed || {
#         echo "Failed for random_seed=$random_seed"
#         continue
#     }
# done

# python main.py --dataset fsmol --encode_method FPaugmentRGB --num_encoder 50 --random_seed 0
