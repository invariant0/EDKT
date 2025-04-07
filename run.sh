# for random_seed in 5 6 7 8 9 10
# do 

# for group_id in 1
# do
#     # python main.py --dataset pQSAR --group_id $group_id --encode_method FP --num_encoder 50 --random_seed 0
#     # python main.py --dataset pQSAR --group_id $group_id --encode_method FPRGB --num_encoder 50 --random_seed 0
#     # python main.py --dataset pQSAR --group_id $group_id --encode_method GraphGCN --random_seed $random_seed
#     # python main.py --dataset pQSAR --group_id $group_id --encode_method GraphGIN --random_seed $random_seed
#     python main.py --dataset pQSAR --group_id $group_id --encode_method GraphGAT --random_seed $random_seed
# done 

# done 
# for random_seed in {20..33}; do
#     python main.py --dataset fsmol --encode_method FPaugmentRGB --num_encoder 2 --random_seed $random_seed 
# done
# for random_seed in {28..36}; do
#     python main.py --dataset fsmol --encode_method FPRGB --num_encoder 2 --random_seed $random_seed 
# done

# python main.py --dataset fsmol --encode_method FPaugmentRGB --num_encoder 50 --random_seed 0
for random_seed in {31..36}; do
    python main.py --dataset fsmol --encode_method GraphGIN --num_encoder 2 --random_seed $random_seed
done
for random_seed in {22..24}; do
    python main.py --dataset fsmol --encode_method GraphSAGE --num_encoder 2 --random_seed $random_seed
done

# for random_seed in {27..54}; do
#     python main.py --dataset fsmol --encode_method GraphGCN --num_encoder 2 --random_seed $random_seed
# done

# for random_seed in {11..21}; do
#     python main.py --dataset fsmol --encode_method GraphGAT --num_encoder 2 --random_seed $random_seed
# done

# for random_seed in {10..30}; do 
#     python main.py --dataset fsmol --encode_method GraphGCN --num_encoder 2 --random_seed $random_seed
# done








