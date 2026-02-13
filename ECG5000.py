import warnings
warnings.filterwarnings("ignore")

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from models.neural_network import CNN1D
from utils.train import train_ASBB
from optimizer.ASBB import StepTunedSGD, AdaptiveBBTuner

def parse_args():
    parser = argparse.ArgumentParser(description='ECG5000 classification problem')
    
    # Dataset and training parameters
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Mini-batch size')
    parser.add_argument('--experiment_name', type=str, default='', help='Experiment name for saving results')
    
    # Optimization parameters
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Step size for gradient updates')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='L2 regularization (ridge)')
    
    # ASBB parameters
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha parameter for ASBB')
    parser.add_argument('--m_tilde', type=float, default=0.01, help='Lower bound for step size in ASBB')
    parser.add_argument('--M_tilde', type=float, default=10.0, help='Upper bound for step size in ASBB')
    parser.add_argument('--window_size', type=int, default=3, help='Window size for updating the parameters')
    parser.add_argument('--initial_tau', type=float, default=0.51, help='Initial tau for ASBB')
        
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()  
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Parameters and Logs
    parameters = {
        'problem': "ECG5000",
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
    }
    
    # Load dataset
    data =[] 
    with open("Data/ECG5000/ECG5000_TRAIN.txt", 'r') as data_file: 
        for line in data_file: 
            # process the line in some way  
            data.append([float(x) for x in line.split(" ") if len(x) > 0])
    data = np.asarray(data)
    trainX, trainY = data[:,1:], np.where(data[:,0] == 1, 0, 1)

    data =[] 
    with open("Data/ECG5000/ECG5000_TEST.txt", 'r') as data_file: 
        for line in data_file: 
            # process the line in some way  
            data.append([float(x) for x in line.split(" ") if len(x) > 0])
    data = np.asarray(data)
    testX, testY = data[:,1:], np.where(data[:,0] == 1, 0, 1)


    # Standardize features
    scaler = RobustScaler()
    trainX = scaler.fit_transform(trainX)
    testX = scaler.transform(testX)

    # Convert to PyTorch tensors
    trainX = torch.FloatTensor(trainX)
    trainY = torch.FloatTensor(trainY)
    testX = torch.FloatTensor(testX)
    testY = torch.FloatTensor(testY)

    # Create DataLoaders
    train_loader = DataLoader(dataset=TensorDataset(trainX, trainY), batch_size=parameters['batch_size'], shuffle=True)
    test_loader = DataLoader(dataset=TensorDataset(testX, testY), batch_size=parameters['batch_size'], shuffle=False)
    
    # Set up and train the model
    model = CNN1D(trainX.shape[1]).to(device)
    # Parameters
    criterion = torch.nn.BCELoss()

    # Load model original weights
    # Setup
    optimizer = StepTunedSGD(
        params=model.parameters(),
        alpha=args.alpha,  
        m_tilde=args.m_tilde,  
        M_tilde=args.M_tilde, 
        weight_decay=parameters['weight_decay'],
        verbose=False
    )

    # Create adaptive tuner
    tuner = AdaptiveBBTuner(
        initial_tau=args.initial_tau,
        initial_beta=0.95,
        tau_range=(0.1, 0.8),
        beta_range=(0.85, 0.99),
        window_size=args.window_size,
        adjustment_rate=0.05,
        verbose=False
    )

    # Train
    train_ASBB(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=parameters['epochs'],
        device=device,
        algorithm="ASBB",
        adaptive_tuner=tuner,
    )
    
if __name__ == "__main__":
    main()