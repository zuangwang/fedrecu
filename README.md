# Federated Learning Experiments

This repository contains code and results for federated learning experiments comparing different algorithms on generated data, CIFAR-10, and CIFAR-100 datasets.

## Project Structure

```
fedrecu/
├── main.py                 # Main execution script
├── models.py               # Neural network model definitions
├── optimizers.py           # Custom optimizers for federated learning
├── trainers.py             # Training algorithms and federated learning methods
├── Code_for_Appendix_A2/   # Code for Appendix_A2
├── data/                   # Dataset storage
│   ├── cifar-10-python.tar.gz
│   ├── cifar-100-python.tar.gz
│   └── [extracted datasets]
├── results/                # Experimental results
│   ├── cifar10_e2000_homFalse_0_L_2_dir_0.1/
│   ├── cifar10_e2000_homFalse_0_L_2_dir_1/
│   ├── cifar10_e2000_homFalse_0_L_2_dir_10/
│   ├── cifar100_e4000_homFalse_0_L_2_dir_0.1/
│   ├── cifar100_e4000_homFalse_0_L_2_dir_1/
│   └── cifar100_e4000_homFalse_0_L_2_dir_10/
```

## Algorithms Compared

The experiments compare four federated learning algorithms
## Datasets and Configurations

### Datasets
- **CIFAR-10**: 10-class image classification
- **CIFAR-100**: 100-class image classification

### Experimental Settings
- **Epochs**: 3000 communication rounds
- **Non-IID Distribution**: Dirichlet distribution with different concentration parameters
  - `dir_0.1`: Highly non-IID (α = 0.1)
  - `dir_1`: Moderately non-IID (α = 1.0)
  - `dir_10`: Mildly non-IID (α = 10.0)

## Usage

### Running Experiments
```bash
python main.py
```
## Results Structure

Each experiment directory contains:
- `{experiment_name}.csv` - Raw experimental data
- `train_accuracy_comparison_{dataset}_dir{α}.png` - Training accuracy plot
- `loss_comparison_{dataset}_dir{α}.png` - Loss progression plot
- `test_accuracy_comparison_{dataset}_dir{α}.png` - Test accuracy plot

## File Format

The CSV files use a block format where each block contains:
```
"Method, hyperparameters"
step0,step1,step2,...
train_accuracy_values
test_accuracy_values
loss_values
```

## Configuration

Key parameters can be adjusted in the respective files:
- Model architecture: `models.py`
- Training algorithms: `trainers.py`
- Optimization settings: `optimizers.py`