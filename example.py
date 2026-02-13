import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from models.neural_network import CNN1D

# %%
# ### Parameters

# %%
# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Parameters 
parameters = {
    'problem': 'ECG5000',
    'batch_size': 128, # Mini-batch size
    'epochs': 10,
    'weight_decay': 1e-4,  # L2 regularization,
    'learning_rate': 1e-3, # Step size for gradient updates,
}
criterion = torch.nn.BCELoss()

# %%
# ### Load data

# %%
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

# %%
# ### Train CNN-1D model

# Set up and train the model
model = CNN1D(trainX.shape[1]).to(device)

from optimizer.StepTunedSGD import StepTunedSGD, AdaptiveBBTuner
from utils.train import train_ASBB

# Setup
optimizer = StepTunedSGD(
    params=model.parameters(),
    alpha=1.0,  
    m_tilde=0.01,  
    M_tilde=10.0,  
    weight_decay=parameters['weight_decay'],
    verbose=False
)

# Create adaptive tuner
tuner = AdaptiveBBTuner(
    initial_tau=0.5,
    initial_beta=0.95,
    tau_range=(0.2, 0.8),
    beta_range=(0.85, 0.99),
    window_size=3,
    adjustment_rate=0.05,
    verbose=False
)

# Train
results = train_ASBB(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    optimizer=optimizer,
    criterion=criterion,
    epochs=parameters['epochs'],
    device=device,
    algorithm="ASBB",
    grad_clip=1.0,
    adaptive_tuner=tuner,
    use_train_loss_for_tuning=False  # Use test loss for adaptation
)


