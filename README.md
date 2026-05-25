# Federated Learning for Healthcare Data Privacy

A PyTorch-based Federated Learning project that demonstrates privacy-preserving machine learning across multiple healthcare-related datasets using **FedAvg aggregation**, **non-IID client simulation**, and a **Differential Privacy-inspired Gaussian noise mechanism**.

---

## Project Overview

Healthcare data is highly sensitive and cannot always be shared directly between hospitals, research labs, or medical institutions. Traditional centralized machine learning requires raw data to be collected on a single server, which creates privacy, security, and regulatory concerns.

This project implements a simulated **Federated Learning (FL)** pipeline where each client trains a model locally on its own data. Instead of sharing raw data, clients send only model updates to a central server. The server then aggregates these updates using the **Federated Averaging (FedAvg)** algorithm.

To add an additional privacy layer, Gaussian noise is applied to client model updates before server aggregation. This demonstrates a Differential Privacy-inspired approach for reducing the exposure of exact local model updates.

---

## Assignment Coverage

| Requirement | Status |
|---|---|
| Implement FL using Python | Completed |
| Use PyTorch/TensorFlow | Completed using PyTorch |
| Use MNIST baseline dataset | Completed |
| Use Heart Disease dataset | Completed |
| Use Diabetes dataset | Completed |
| Simulate 3–5 clients | Completed with 3 clients |
| Use non-IID data distribution | Completed |
| Add one research extension | Differential Privacy-inspired Gaussian noise |
| Show accuracy comparison | Completed |
| Show communication overhead | Completed |
| Show convergence plots | Completed |

---

## Key Features

- Federated Learning implementation using **PyTorch**
- Simulation of **3 federated clients**
- Non-IID data distribution across clients
- FedAvg-based global model aggregation
- Differential Privacy-inspired Gaussian noise on client updates
- Supports both classification and regression tasks
- Tracks round-wise loss, accuracy, RMSE, and communication overhead
- Generates convergence plots and final comparison results
- Saves trained global model checkpoints

---

## Datasets Used

| Dataset | Task Type | Metric | Non-IID Split Strategy |
|---|---|---|---|
| MNIST | Multi-class Classification | Accuracy | Label-skew split |
| Heart Disease | Binary Classification | Accuracy | Class-ratio skew |
| Diabetes | Regression | RMSE | Target-range quantile split |

---

## Federated Learning Configuration

| Parameter | Value |
|---|---|
| Number of Clients | 3 |
| Communication Rounds | 10 |
| Local Epochs per Round | 3 |
| Aggregation Algorithm | FedAvg |
| DP Noise Standard Deviation | 0.001 |
| Framework | PyTorch |

---

## Final Results

| Dataset | Task | Final Metric | Final Loss | Communication Overhead |
|---|---|---:|---:|---:|
| MNIST | Classification | 78.71% Accuracy | 0.0196 | 25.04 MB |
| Heart Disease | Binary Classification | 63.93% Accuracy | 0.6484 | 0.73 MB |
| Diabetes | Regression | RMSE 64.89 | 0.3401 | 0.69 MB |
| **Total** | — | — | — | **26.46 MB** |

---

## System Architecture

```text
Dataset
   |
   v
Data Preprocessing
   |
   v
Non-IID Client Split
   |
   +-------------------+-------------------+-------------------+
   |                   |                   |                   |
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
```

---

## Project Structure

```text
federated_healthcare/
│
├── main.py                 # Main entry point for running all experiments
├── model.py                # Model definitions for MNIST, Heart Disease, and Diabetes
├── client.py               # Federated client logic and local training
├── server.py               # FedAvg aggregation logic
├── utils.py                # Data loading, preprocessing, and non-IID splitting
├── plots.py                # Plot generation and result visualization
│
├── data/
│   └── heart.csv           # Heart Disease dataset file
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
```

---

## Model Architectures

### MNIST Model

The MNIST model is a simple Multi-Layer Perceptron used for baseline multi-class classification.

```text
Input: 784 features
Linear(784 → 128)
ReLU
Linear(128 → 64)
ReLU
Linear(64 → 10)
```

- Task: Multi-class classification
- Loss Function: CrossEntropyLoss
- Output Classes: 10 digits

---

### Heart Disease Model

The Heart Disease model is a binary classifier designed for tabular healthcare data.

```text
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
```

- Task: Binary classification
- Loss Function: BCEWithLogitsLoss
- Preprocessing: StandardScaler
- Stabilization: LayerNorm and Xavier initialization

---

### Diabetes Model

The Diabetes model is a regression network used to predict disease progression.

```text
Input: 10 features
Linear(10 → 64)
LayerNorm
ReLU
Linear(64 → 32)
LayerNorm
ReLU
Linear(32 → 1)
```

- Task: Regression
- Loss Function: MSELoss
- Target values are normalized during training
- RMSE is reported on the original target scale

---

## Non-IID Client Simulation

In real-world healthcare systems, different hospitals may have different patient demographics, disease distributions, and data patterns. To simulate this, the project uses non-IID data splits.

### MNIST

MNIST uses label-skew distribution, where different clients receive different digit groups.

| Client | Data Distribution |
|---|---|
| Client 1 | Digits 0–3 |
| Client 2 | Digits 4–6 |
| Client 3 | Digits 7–9 |

### Heart Disease

Heart Disease uses class-ratio skew. Each client receives a different proportion of positive and negative cases.

### Diabetes

Diabetes uses a target-range quantile split. Each client receives samples from different disease progression ranges.

---

## Differential Privacy-Inspired Mechanism

A Gaussian noise mechanism is applied to client model parameters before the updates are sent to the central server.

```python
if self.dp_noise_std > 0.0:
    with torch.no_grad():
        for param in self.model.parameters():
            noise = torch.randn_like(param) * self.dp_noise_std
            param.add_(noise)
```

| Property | Value |
|---|---|
| Noise Type | Gaussian |
| Noise Standard Deviation | 0.001 |
| Applied To | Client model parameters |
| Applied Before | Server aggregation |
| Aggregation | FedAvg |

> Note: This project demonstrates a Differential Privacy-inspired privacy mechanism. A production-grade DP system would require formal epsilon-delta privacy accounting, secure aggregation, and stronger deployment-level security controls.

---

## Communication Overhead

Communication overhead is calculated as:

```text
Communication Overhead = 2 × Number of Clients × Model Size × Number of Rounds
```

The factor of `2` represents:

1. Server-to-client global model download
2. Client-to-server local model update upload

MNIST has the highest communication overhead because its model has more parameters compared to the Heart Disease and Diabetes models.

---

## Results and Plots

The project generates the following result files inside the `results/` folder:

| File | Description |
|---|---|
| `mnist_convergence.png` | MNIST loss and accuracy convergence |
| `heart_convergence.png` | Heart Disease loss and accuracy convergence |
| `diabetes_convergence.png` | Diabetes loss and RMSE convergence |
| `communication_overhead.png` | Communication overhead comparison |
| `comparison_table.png` | Final results comparison table |
| `mnist_loss.png` | MNIST loss curve |
| `mnist_accuracy_pct.png` | MNIST accuracy curve |
| `heart_loss.png` | Heart Disease loss curve |
| `heart_accuracy_pct.png` | Heart Disease accuracy curve |
| `diabetes_loss.png` | Diabetes loss curve |
| `diabetes_rmse.png` | Diabetes RMSE curve |

---

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Pranavarya18/federated-learning-healthcare.git
cd federated-learning-healthcare
```

### 2. Create a Virtual Environment

For Windows:

```bash
python -m venv venv
venv\Scripts\activate.bat
```

For macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If needed, dependencies can also be installed manually:

```bash
pip install torch torchvision scikit-learn pandas numpy matplotlib
```

### 4. Run the Project

```bash
python main.py
```

---

## Expected Output

After running `main.py`, the terminal displays training progress for all three datasets.

```text
FEDERATED LEARNING — HEALTHCARE PROJECT
Datasets: MNIST + Heart Disease + Diabetes
Clients: 3 | Rounds: 10 | DP: 0.001

FEDERATED LEARNING — MNIST
Round 1/10 | Loss: ... | Accuracy: ... | Overhead: ...

FEDERATED LEARNING — HEART DISEASE
Round 1/10 | Loss: ... | Accuracy: ... | Overhead: ...

FEDERATED LEARNING — DIABETES
Round 1/10 | Loss: ... | RMSE: ... | Overhead: ...

FINAL RESULTS SUMMARY
MNIST Accuracy
Heart Disease Accuracy
Diabetes RMSE
Total Communication Overhead
```

---

## Evaluation Metrics

| Task | Metrics Used |
|---|---|
| MNIST Classification | Accuracy, CrossEntropyLoss |
| Heart Disease Classification | Accuracy, BCEWithLogitsLoss |
| Diabetes Regression | RMSE, MSELoss |
| Federated System | Communication Overhead, Convergence |

---

## Challenges Addressed

| Challenge | Solution Applied |
|---|---|
| Heart model accuracy stuck around 50% | Applied StandardScaler and BCEWithLogitsLoss |
| Sigmoid saturation in binary classification | Used raw logits with BCEWithLogitsLoss |
| Small healthcare dataset instability | Used LayerNorm instead of BatchNorm |
| Diabetes regression instability | Normalized target values before training |
| Non-IID client drift | Used multiple FedAvg communication rounds |
| Privacy risk in client updates | Added Gaussian noise before aggregation |

---

## Limitations

- The clients are simulated on a single machine.
- Only 3 clients are used.
- Differential Privacy is implemented as a simplified Gaussian noise mechanism.
- Formal epsilon-delta privacy accounting is not implemented.
- Secure aggregation is not implemented.
- Heart Disease accuracy is moderate due to small dataset size and non-IID class skew.
- Real-world healthcare deployment would require stronger validation, compliance checks, and privacy guarantees.

---

## Future Improvements

- Add secure aggregation.
- Add formal epsilon-delta Differential Privacy accounting.
- Use weighted FedAvg based on client dataset size.
- Increase the number of clients.
- Evaluate on larger real-world healthcare datasets.
- Add model compression to reduce communication overhead.
- Compare federated learning with centralized training.
- Add blockchain-based logging for auditability.
- Build a dashboard for monitoring FL training rounds.

---

## Key Learnings

This project demonstrates:

- How Federated Learning supports privacy-preserving machine learning
- How non-IID data affects convergence
- How FedAvg combines client models into a global model
- How Gaussian noise can reduce exposure of exact client updates
- How communication overhead can be tracked in FL systems
- How FL can be applied to both classification and regression tasks

---

## References

1. H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, “Communication-Efficient Learning of Deep Networks from Decentralized Data,” AISTATS, 2017.
2. C. Dwork and A. Roth, “The Algorithmic Foundations of Differential Privacy,” Foundations and Trends in Theoretical Computer Science, 2014.
3. M. Abadi et al., “Deep Learning with Differential Privacy,” ACM CCS, 2016.
4. A. Paszke et al., “PyTorch: An Imperative Style, High-Performance Deep Learning Library,” NeurIPS, 2019.
5. D. Dua and C. Graff, “UCI Machine Learning Repository,” University of California, Irvine, 2019.
6. Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-Based Learning Applied to Document Recognition,” Proceedings of the IEEE, 1998.

---

## Author

**Pranav Arya**  
Federated Learning for Healthcare Dataset  

---

## License

This project is developed for academic and educational purposes.