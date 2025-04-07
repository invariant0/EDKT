EDKT: Ensemble Diversified Kernel Transfer for Bio-Activity Prediction
Please cite our work if you find it helpful!

This repository contains the official implementation of the EDKT model for bio-activity prediction, focusing on training diversified deep kernel functions.

Table of Contents
Installation
Training the Model
Generating Predictions
Performance Analysis
Citation
Installation
Clone the repository and install dependencies:

bash

Copy
git clone [your-repo-url]  
cd EDKT  
pip install -r requirements.txt  # Add a requirements.txt if applicable
Training the Model
To train a diversified DKT model with distinct settings, run:

bash

Copy
python main.py \
  --dataset fsmol \
  --encode_method {encode_method} \
  --num_encoder {num_encoder} \
  --random_seed {random_seed}
Parameters
encode_method: Choose from:
GraphGAT, GraphSAGE, GraphGIN, GraphGCN, FP, FPRBF, FPaugment, FPaugmentRBF
num_encoder: Adjust based on GPU capacity. For graph encoders (e.g., GAT, GIN), use 2 if your GPU has <24 GB RAM.
random_seed: Controls stochastic initialization.
Trained models are automatically saved in the Model_for_publication directory.

Generating Predictions
Generate ensemble predictions for different datasets:

FS-Mol Dataset
bash

Copy
python prediction_gen_fsmol.py
pQSAR-ChEMBL Dataset
bash

Copy
python prediction_gen_pqsar.py
Results are saved in the Result_for_publication folder.

Performance Analysis
The following analyses are supported:

Shapley Value Analysis
Quantifies feature importance for interpretability.
Bias-Variance Decomposition
Diagnoses model performance trade-offs.
Calibration Plot Analysis
Evaluates prediction confidence reliability.