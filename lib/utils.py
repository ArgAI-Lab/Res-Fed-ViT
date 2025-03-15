import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from scipy import sparse as sp

import numpy as np
import networkx as nx
import copy




import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from collections import defaultdict, Counter

def split_train_loader(train_loader, num_clients, non_iid=True, alpha=0.5):
    """
    Splits a DataLoader's underlying dataset among clients.
    
    For IID splitting, it simply splits the dataset equally.
    For non-IID splitting, it partitions data based on a Dirichlet distribution.
    Also prints the number of samples each client receives from each class.
    
    Args:
        train_loader: DataLoader object for the training dataset.
        num_clients: Number of clients to split the dataset into.
        non_iid: Boolean flag; if True, perform non-IID (Dirichlet) split.
        alpha: Dirichlet concentration parameter (used only if non_iid is True).
        
    Returns:
        A list of DataLoader objects, one for each client.
    """
    dataset = train_loader.dataset
    batch_size = train_loader.batch_size

    # We'll need to extract the labels. We assume each dataset sample is (data, label)
    labels = [dataset[i][1] for i in range(len(dataset))]
    labels = np.array(labels)
    num_classes = len(np.unique(labels))
    
    client_indices = {}
    
    if not non_iid:
        # IID splitting: simply use random_split
        total_size = len(dataset)
        split_sizes = [total_size // num_clients + (1 if x < total_size % num_clients else 0) for x in range(num_clients)]
        subsets = torch.utils.data.random_split(dataset, split_sizes)
        # Collect indices for printing distribution
        start_idx = 0
        for client_id, subset in enumerate(subsets):
            # Random split doesn't provide indices, so we reconstruct approximate indices
            # For printing purposes, we count labels by iterating through the subset.
            client_indices[client_id] = [i for i in range(start_idx, start_idx + len(subset))]
            start_idx += len(subset)
        client_loaders = [DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=8) for subset in subsets]
    else:
        # Non-IID splitting using Dirichlet distribution
        client_indices = {i: [] for i in range(num_clients)}
        # For each class, distribute its indices among clients using Dirichlet distribution
        for c in range(num_classes):
            idx_c = np.where(labels == c)[0]
            np.random.shuffle(idx_c)
            # Sample proportions for this class among clients from Dirichlet distribution
            proportions = np.random.dirichlet(alpha * np.ones(num_clients))
            # Determine the number of samples for each client for class c
            proportions = (proportions * len(idx_c)).astype(int)
            # Adjust rounding issues to ensure the total equals len(idx_c)
            diff = len(idx_c) - np.sum(proportions)
            for i in range(diff):
                proportions[i % num_clients] += 1
            # Split indices for class c according to computed proportions
            start = 0
            for client_id, count in enumerate(proportions):
                client_indices[client_id].extend(idx_c[start:start+count].tolist())
                start += count

        client_loaders = []
        for client_id in range(num_clients):
            client_subset = Subset(dataset, client_indices[client_id])
            loader = DataLoader(client_subset, batch_size=batch_size, shuffle=True, num_workers=8)
            client_loaders.append(loader)
    
    # Print class distributions for each client
    print("Client class distributions:")
    for client_id, indices in client_indices.items():
        client_labels = [labels[i] for i in indices]
        counter = Counter(client_labels)
        distribution = ", ".join([f"class {cls}: {counter.get(cls, 0)}" for cls in range(num_classes)])
        print(f"Client {client_id}: {distribution}")
        
    return client_loaders




# Assuming other necessary imports and device configuration are done as in your existing setup

def federated_train_step(args,model,global_model, train_loader, optimizer, criterion,attack, device):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        if attack:
            adv_images = attack(images, labels)
            outputs = model(adv_images)
        else:
            outputs = model(images)
        loss = criterion(outputs, labels)
        # Add the proximal term
        if args.alg == "fedprox":
            for param, param_global in zip(model.parameters(), global_model.parameters()):
                loss += (args.mu / 2) * torch.norm(param - param_global, p=2)**2

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, test_loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return  100 * correct / total



def average_weights(args ,weights, data_sizes):
    """
    Averages the model weights of clients based on the amount of data each client has.

    Parameters:
    - weights: List of state_dicts from each client's model.
    - data_sizes: List of data sizes for each client.

    Returns:
    - average: Averaged state_dict.
    """
    total_data = sum(data_sizes)
    average = copy.deepcopy(weights[0])

    for key in average.keys():
        if "num_batches_tracked" not in key:
            average[key] *= data_sizes[0] / total_data
            for i in range(1, len(weights)):
                average[key] += weights[i][key] * (data_sizes[i] / total_data)
        else:
            average[key] = torch.zeros_like(average[key])
    return average
