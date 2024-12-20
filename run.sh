for random_seed in 14 15 16 17 18 19 20 21 22 23 24 25
do 

for group_id in 1
do
    # python main.py --dataset pQSAR --group_id $group_id --encode_method FP --num_encoder 50 --random_seed 0
    # python main.py --dataset pQSAR --group_id $group_id --encode_method FPRGB --num_encoder 50 --random_seed 0
    # python main.py --dataset pQSAR --group_id $group_id --encode_method GraphGCN --random_seed $random_seed
    # python main.py --dataset pQSAR --group_id $group_id --encode_method GraphGIN --random_seed $random_seed
    python main.py --dataset pQSAR --group_id $group_id --encode_method GraphSAGE --random_seed $random_seed
done 

done 

# for random_seed in {1..10}; do
#     python main.py --dataset fsmol --encode_method FPRGB --num_encoder 50 --random_seed $random_seed || {
#         echo "Failed for random_seed=$random_seed"
#         continue
#     }
# done

# python main.py --dataset fsmol --encode_method FPaugmentRGB --num_encoder 50 --random_seed 0

# for random_seed in {1..12}; do
#     python main.py --dataset fsmol --encode_method GraphSAGE --num_encoder 2 --random_seed $random_seed
#     python main.py --dataset fsmol --encode_method GraphGCN --num_encoder 2 --random_seed $random_seed
# done

# for random_seed in {10..30}; do 
#     python main.py --dataset fsmol --encode_method GraphGCN --num_encoder 2 --random_seed $random_seed
# done








