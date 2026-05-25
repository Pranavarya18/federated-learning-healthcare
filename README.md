Federated Learning for Healthcare Data Privacy
A PyTorch-based Federated Learning project that demonstrates privacy-preserving machine learning across multiple healthcare-related datasets using FedAvg, Non-IID client simulation, and a Differential Privacy-inspired Gaussian noise mechanism.

Overview
Healthcare data is highly sensitive and cannot always be shared directly between hospitals, labs, or institutions. Traditional centralized machine learning requires all data to be collected in one place, which creates privacy, security, and regulatory concerns.

This project implements a simulated Federated Learning (FL) pipeline where each client trains locally on its own data, and only model updates are sent to the central server. The server aggregates these updates using the Federated Averaging (FedAvg) algorithm.

To add an extra privacy layer, Gaussian noise is applied to client model updates before aggregation.

The project covers three datasets:

MNIST → Baseline multi-class classification task
Heart Disease → Binary healthcare classification task
Diabetes → Healthcare regression task
Key Features
Federated Learning implementation using PyTorch
Simulation of 3 federated clients
Non-IID data distribution across clients
FedAvg server-side aggregation
Differential Privacy-inspired update noise
Classification + Regression support
Communication overhead calculation
Round-wise loss, accuracy, and RMSE tracking
Result plots and saved model checkpoints
Tech Stack
Component	Technology
Programming Language	Python
Deep Learning Framework	PyTorch
ML Utilities	scikit-learn
Data Processing	NumPy, Pandas
Visualization	Matplotlib
Aggregation	FedAvg
Privacy Extension	Gaussian Noise on Client Updates
Datasets
Dataset	Task Type	Output Metric	Non-IID Split Strategy
MNIST	Multi-class Classification	Accuracy	Label-skew split
Heart Disease	Binary Classification	Accuracy	Class-ratio skew
Diabetes	Regression	RMSE	Target-range quantile split
Federated Learning Setup
Parameter	Value
Number of Clients	3
Communication Rounds	10
Local Epochs per Round	3
Aggregation Algorithm	FedAvg
DP Noise Standard Deviation	0.001
Framework	PyTorch
Final Results
Dataset	Task	Final Metric	Final Loss	Communication Overhead
MNIST	Classification	78.71% Accuracy	0.0196	25.04 MB
Heart Disease	Binary Classification	63.93% Accuracy	0.6484	0.73 MB
Diabetes	Regression	RMSE 64.89	0.3401	0.69 MB
Total	—	—	—	26.46 MB
Project Architecture
Dataset
   |
   v
Preprocessing
   |
   v
Non-IID Client Split
   |
   +-------------------+-------------------+-------------------+
   |                   |                   |
   v                   v                   v
Client 1            Client 2            Client 3
Local Training      Local Training      Local Training
   |                   |                   |
   v                   v                   v
DP Noise Addition   DP Noise Addition   DP Noise Addition
   |                   |                   |
   +-------------------+-------------------+
                       |
                       v
                Central Server
                       |
                       v
               FedAvg Aggregation
                       |
                       v
             Global Model Evaluation
Model Architectures
1. MNIST Model
Input: 784 features

Linear(784 → 128)
ReLU
Linear(128 → 64)
ReLU
Linear(64 → 10)
2. Heart Disease Model
Input: 13 features

Linear(13 → 64)
LayerNorm
ReLU
Dropout
Linear(64 → 32)
LayerNorm
ReLU
Dropout
Linear(32 → 1)
3. Diabetes Model
Input: 10 features

Linear(10 → 64)
LayerNorm
ReLU
Linear(64 → 32)
LayerNorm
ReLU
Linear(32 → 1)
Project Structure
federated_healthcare/
│
├── main.py                  # Main entry point for running all experiments
├── model.py                 # Model definitions
├── client.py                # Federated client logic and local training
├── server.py                # FedAvg aggregation logic
├── utils.py                 # Data loading, preprocessing, Non-IID splitting
├── plots.py                 # Result visualization utilities
│
├── data/
│   └── heart.csv            # Heart Disease dataset file
│
├── results/
│   ├── mnist_loss.png
│   ├── mnist_accuracy_pct.png
│   ├── mnist_convergence.png
│   ├── heart_loss.png
│   ├── heart_accuracy_pct.png
│   ├── heart_convergence.png
│   ├── diabetes_loss.png
│   ├── diabetes_rmse.png
│   ├── diabetes_convergence.png
│   ├── communication_overhead.png
│   ├── comparison_table.png
│   ├── mnist_model.pth
│   ├── heart_model.pth
│   └── diabetes_model.pth
│
├── requirements.txt
└── README.md
Privacy Mechanism
To simulate lightweight privacy preservation:

Gaussian noise is added to client model updates before server aggregation.
This reduces exposure of exact local learned patterns.
Mimics Differential Privacy-inspired Federated Learning.
Evaluation Metrics
Classification Tasks (MNIST + Heart Disease)
Accuracy
Cross-Entropy Loss / BCE Loss
Regression Task (Diabetes)
RMSE (Root Mean Square Error)
MSE Loss
Federated Performance Metrics
Communication Overhead
Round-wise Convergence
Training Loss
Key Learnings
This project demonstrates:

Federated Learning for privacy-preserving AI
Distributed healthcare collaboration simulation
Handling Non-IID client datasets
FedAvg-based aggregation
Trade-off between privacy and performance
Communication cost monitoring
Multi-task FL (Classification + Regression)
Conclusion
This project successfully demonstrates a privacy-preserving Federated Learning pipeline for healthcare-related tasks using FedAvg, Non-IID client simulation, and Gaussian noise-based privacy enhancement.

Despite heterogeneous datasets and decentralized training, the system achieved stable convergence and meaningful predictive performance across both classification and regression tasks.

It highlights how Federated Learning can serve as an effective and scalable solution for secure healthcare AI systems.

