import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
from src.models import SimpleCNN

class Client:
    def __init__(self, client_id, dataset, device, dataset_name, batch_size=50, lr=0.01):
        self.id = client_id
        self.dataset = dataset
        self.device = device
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.lr = lr
        
        # Heterogeneity (Fast vs Slow)
        if client_id < 60: 
            speed_val = np.random.lognormal(4.1, 0.2) 
            self.base_compute_factor = np.clip(speed_val, 40.0, 90.0)
        else:
            speed_val = np.random.lognormal(5.2, 0.2)
            self.base_compute_factor = np.clip(speed_val, 150.0, 300.0)
        self.base_upload_time = random.uniform(5.0, 10.0)

    def estimate_time_with_epochs(self, epochs):
        system_noise = random.uniform(0.9, 1.1)
        data_volume = len(self.dataset)
        compute_time = data_volume * self.base_compute_factor * 0.0005 * epochs 
        return (compute_time + self.base_upload_time) * system_noise

    def train(self, global_weights, epochs):
        # CIFAR-10은 무조건 3채널
        model = SimpleCNN(num_channels=3).to(self.device)
        model.load_state_dict(global_weights)
        model.train()
        
        optimizer = optim.SGD(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        executed_epochs = 0
        for _ in range(epochs):
            for data, target in self.loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(data), target)
                loss.backward()
                optimizer.step()
            executed_epochs += 1
            
        return model.state_dict(), executed_epochs