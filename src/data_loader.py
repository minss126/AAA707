import numpy as np
import random
from torchvision import datasets, transforms
from torch.utils.data import Subset

def get_dataset_realistic_noniid(dataset_name, num_clients):
    """
    CIFAR-10 Resource-Dependent Non-IID Data Generation
    """
    print(f">>> [Data] Loading CIFAR-10 with Resource-Dependent Non-IID Bias...")
    
    # CIFAR-10 전용 Transform (3채널)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    data_dir = './data'

    # 무조건 CIFAR-10 다운로드
    train_data = datasets.CIFAR10(data_dir, train=True, download=True, transform=tf)
    test_data = datasets.CIFAR10(data_dir, train=False, transform=tf)
    
    # Non-IID 분배 로직
    targets = np.array(train_data.targets)
    idx_major = np.where(targets < 5)[0]  
    idx_minor = np.where(targets >= 5)[0] 
    np.random.shuffle(idx_major)
    np.random.shuffle(idx_minor)
    
    ptr_major = 0
    ptr_minor = 0
    client_datasets = []
    
    for i in range(num_clients):
        idx_list = []
        samples = 300
        
        # Fast Clients (Major Data Holders)
        if i < 60: 
            n_main = int(samples * 0.95)
            n_sub = samples - n_main
            if ptr_major + n_main > len(idx_major): ptr_major = 0
            idx_list.extend(idx_major[ptr_major : ptr_major+n_main])
            ptr_major += n_main
            if ptr_minor + n_sub > len(idx_minor): ptr_minor = 0
            idx_list.extend(idx_minor[ptr_minor : ptr_minor+n_sub])
            ptr_minor += n_sub
            
        # Slow Clients (Minor Data Holders)
        else:
            n_main = int(samples * 0.95)
            n_sub = samples - n_main
            if ptr_minor + n_main > len(idx_minor): ptr_minor = 0
            idx_list.extend(idx_minor[ptr_minor : ptr_minor+n_main])
            ptr_minor += n_main
            if ptr_major + n_sub > len(idx_major): ptr_major = 0
            idx_list.extend(idx_major[ptr_major : ptr_major+n_sub])
            ptr_major += n_sub
            
        random.shuffle(idx_list)
        client_datasets.append(Subset(train_data, idx_list))
        
    return client_datasets, test_data