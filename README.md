# EDKT

please cite if you find help!

This is the code implementaion for EDKT model for bio-activity prediction

# Enviroment setup 

```bash  
git clone https://github.com/invariant0/EDKT.git
cd EDKT  
pip install -r requirements.txt  
``` 

# Training stage

The training for each DKT model with distinct setting can be run by following command line:

```bash 
python main.py --dataset fsmol --encode_method ENCODER --num_encoder ENCODER_NUM --random_seed R_SEED
``` 

`ENCODER` could be either one of them in :{GraphGAT, GraphSAGE, GraphGIN, GraphGCN, FP, FPRBF, FPaugment, FPaugmentRBF}

`ENCODER_NUM` could be whatever that maximize your GPU utilization without OOM error, for graph encoding methods 2 is recomended if your GPU has less than 24 GB RAM.

`R_SEED` specify the different stochastic initialization of the model training. 

After finishing training, the model will be saved in Model_for_publication folder

# Inference stage

result for fs-mol datasets can be gathered by running the following script
```bash
python prediction_gen_fsmol.py
```
result for pQSAR-ChEMBL datasets can be gathered by running the following script
```bash
python prediction_gen_pqsar.py
```
prediction result will be saved in Result_for_publication folder 

# Performance analysis 


# Bias-Variance decomposition analysis


# Calibration plot analysis 







