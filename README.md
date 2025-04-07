# EDKT

please cite if you find help!

This is the code implementaion for EDKT model for bio-activity prediction

# Enviroment setup 

```bash  
git clone https://github.com/invariant0/EDKT.git
cd EDKT  
pip install -r requirements.txt  
``` 

# Training a diversified deep kernel function

The training for each DKT model with distinct setting can be run by following command line:

```bash 
python main.py --dataset fsmol --encode_method {encode_method} --num_encoder {num_encoder} --random_seed {random_seed}
``` 

encode_method could be either one of them in :{GraphGAT, GraphSAGE, GraphGIN, GraphGCN, FP, FPRBF, FPaugment, FPaugmentRBF}

num_encoder could be whatever that maximize your GPU utilization without OOM error, for graph encoding methods 2 is recomended if your GPU has less than 24 GB RAM.

random_seed specify the different stochastic initialization of the model training. 

After finishing training, the model will be saved in Model_for_publication folder

# Generate prediction result using ensemble from trained models

result for fs-mol datasets can be gathered by running the following script
```bash
python prediction_gen_fsmol.py
```
result for pQSAR-ChEMBL datasets can be gathered by running the following script
```bash
python prediction_gen_.py
```
prediction result will be saved in Result_for_publication folder 

# Performance analysis 


# Bias-Variance decomposition analysis



# Calibration plot analysis 







