PROJECT README

Project Overview  
This repository implements a federated learning framework that supports both deep learning models (PyTorch + timm) and traditional machine-learning models (scikit-learn). The framework is designed for experimentation with decentralized training, model aggregation, client heterogeneity, and comparison of neural and non-neural classifiers within a unified engineering environment. It provides modular code capable of running Federated Averaging (FedAvg), FedProx, traditional-model federated ensemble, and linear-aggregation strategies, as well as complete visualization and evaluation utilities.

Key Capabilities  
Deep Learning Support — The system incorporates a flexible model wrapper that allows any timm backbone to be used as a federated model. Each model is constructed with a classifier head sized automatically from the backbone’s output. VGG-16 is handled via convolutional feature extraction to avoid overly large fully connected layers (see models.py).  
Traditional Machine-Learning Support — The project contains multiple scikit-learn pipelines including SVM (RBF, Linear-SVC + calibration, KernelPCA variants), Random Forest, Naïve Bayes, and K-Nearest Neighbors. Each pipeline integrates preprocessing steps, including variance thresholding, scaling, PCA, and optional HOG and color histogram features (see ml_pipeline.py and ml_features.py).  
Federated Learning Algorithms — The federated training utilities implement FedAvg, FedProx, federated ensemble for traditional models, and linear federated aggregation. These implementations manage client selection, model distribution, local training, server-side evaluation, and aggregation of weights or probability outputs (see federated_utils.py).  
Experiment & Evaluation Tools — The system includes tools for computing AUC, ROC curves, probability distributions, and aggregating client test results. It generates confusion matrices, multi-class ROC visualisations, federated accuracy curves, per-client performance panels, and server-client overlays—suitable for both deep and traditional models (see experiment_utils.py and plotting.py).

Project Structure  
project/  
 client.py  
 server.py  
 config.py  
 data_prep.py  
 data_utils.py  
 models.py  
 ml_features.py  
 ml_pipeline.py  
 federated_utils.py  
 experiment_utils.py  
 plotting.py  
 main.py  
 requirements.txt

Client modules define local training logic for deep learning and FedProx. Server modules coordinate global aggregation, validation, and final evaluation. The configuration module defines dataset paths, selected models, and global hyperparameters. Data preparation and utilities handle dataset construction and tensor preprocessing. The main script orchestrates data loading, client dataset partitioning, federated training execution, and model evaluation.

Running the System  
1. Install dependencies using the provided requirements file (see requirements.txt).  
2. Prepare the datasets and adjust path settings inside config.py.  
3. Execute the primary experiment pipeline via:  
   python main.py  
   This script handles dataset loading, client partitioning, model selection (deep vs. traditional), federated training, and the automatic generation of plots and metrics.

Model Visualisation and Performance Analysis  
All performance figures are generated automatically during training. They include server and client ROC curves, confusion matrices, per-client learning curves, and consolidated comparisons across clients and server nodes. The visualisation utilities produce high-quality, publication-ready plots suitable for academic reports or professional documentation (see plotting.py).

System Design Summary  
This repository illustrates a complete engineering workflow for federated machine-learning research. The design emphasises modularity, reproducibility, and extensibility.  
• Any timm-compatible backbone may be used as a federated model.  
• Traditional models may be federated via probability-level or weight-level aggregation.  
• All computations, metrics, and visualisations are automated.  
• The codebase separates concerns across model construction, federated algorithm implementation, experiment management, and plotting subsystems.

This project demonstrates federated learning systems engineering, deep-learning integration, model interoperability, and large-scale experimentation.

Author  
Ticauris “Ti” Stokes  

