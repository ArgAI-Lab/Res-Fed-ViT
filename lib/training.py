import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import torch
import pandas as pd
import os


def run_experiment(model, train_loader, test_loader, attack ,device,local_epoch):

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    epoch_data = []

    for epoch in range(local_epoch):  # Define num_epochs as needed
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            # Generate adversarial examples
            if attack:
                adv_images = attack(images, labels)
                outputs = model(adv_images)
            else:
                outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total

        epoch_data.append({
            'epoch': epoch + 1,
            'accuracy': accuracy,
            'loss': loss.item()
        })
        print(f"Epoch {epoch+1}: Accuracy {accuracy:.2f}%")

    df = pd.DataFrame(epoch_data)
    return df,model


        

def run_federated_experiment(args,model, federated_train_loaders, test_loader, device, attack=None):
    criterion = torch.nn.CrossEntropyLoss()
    global_model = model.to(device)
    local_models= []

    client_results = {
        'epoch': [],
        'client_id': [],
        'client_loss': [],
        'client_accuracy': []
    }
    global_results = {
        'epoch': [],
        'global_loss': [],
        'global_accuracy': [],
        'client_avg_accuracy': []
    }
    data_sizes = [len(loader.dataset) for loader in federated_train_loaders]
    total_data = sum(data_sizes)

    for i in range(len(data_sizes)):
        data_sizes[i]= data_sizes[i] / total_data



    for client_id, train_loader in enumerate(federated_train_loaders):
        local_models.append(copy.deepcopy(global_model))
    


    for epoch in range(args.epoch):  # Define num_epochs as needed
        client_losses = []
        client_accs = []
        # Randomly select 80% of clients for this epoch
        num_selected = max(1, int(0.8 * len(data_sizes)))

        selected_clients = np.random.choice(range(len(data_sizes)), size=num_selected, replace=False)

        print(f"\nEpoch {epoch+1}: Selected clients: {selected_clients}")

        selected_models = []
        selected_data_sizes = []


        # Train each client model and evaluate its accuracy
        for client_id in selected_clients:
            train_loader = federated_train_loaders[client_id]

            optimizer = torch.optim.AdamW(local_models[client_id].parameters(), lr=args.lr, weight_decay=args.weight_decay)
            local_loss = federated_train_step(args,local_models[client_id],global_model, train_loader, optimizer, criterion,attack, device)
            client_losses.append(local_loss)

            # Evaluate the local model
            local_accuracy = evaluate(local_models[client_id], test_loader, device)
            client_accs.append(local_accuracy)
            print(f"Epoch :{epoch+1} client_id:{client_id + 1} client Test Accuracy: {local_accuracy:.2f}%, client Loss: {local_loss:.4f}")

            # Save per-client results
            client_results['epoch'].append(epoch + 1)
            client_results['client_id'].append(client_id + 1)
            client_results['client_loss'].append(local_loss)
            client_results['client_accuracy'].append(local_accuracy)
                        
            selected_models.append(copy.deepcopy(local_models[client_id]))
            selected_data_sizes.append(data_sizes[client_id])
        # Aggregate weights and calculate global model's performance
        global_model, local_models = average_weights_v2(args, global_model, selected_models, selected_data_sizes,local_models)


        # Evaluate global model for accuracy
        global_accuracy = global_test_accuracy(args, global_model,local_models, test_loader, device)

        # Calculate global loss (if needed, you may want to run it through a loss function using a dataset)
        global_loss = sum(client_losses) / len(client_losses)  # Example averaging client losses for global loss approximation

        # Calculate the average client accuracy
        client_avg_accuracy = sum(client_accs) / len(client_accs)

        global_results['epoch'].append(epoch + 1)
        global_results['global_loss'].append(global_loss)
        global_results['global_accuracy'].append(global_accuracy)
        global_results['client_avg_accuracy'].append(client_avg_accuracy)

        print(f"Epoch {epoch+1}: Global Test Accuracy: {global_accuracy:.2f}%, Global Loss: {global_loss:.4f}")
        
    # Save model:
    save_model(args, global_model, local_models)

    # Convert results to pandas DataFrames
    client_df = pd.DataFrame(client_results)
    global_df = pd.DataFrame(global_results)
    
    return client_df, global_df


# def run_federated_experiment(args,model, federated_train_loaders, test_loader, device, attack=None):
#     criterion = torch.nn.CrossEntropyLoss()
#     global_model = model.to(device)
#     local_models= []

#     client_results = {
#         'epoch': [],
#         'client_id': [],
#         'client_loss': [],
#         'client_accuracy': []
#     }
#     global_results = {
#         'epoch': [],
#         'global_loss': [],
#         'global_accuracy': [],
#         'client_avg_accuracy': []
#     }
#     data_sizes = [len(loader.dataset) for loader in federated_train_loaders]
#     total_data = sum(data_sizes)
#     for i in range(len(data_sizes)):
#         data_sizes[i]= data_sizes[i] / total_data

#     for client_id, train_loader in enumerate(federated_train_loaders):
#         local_models.append(copy.deepcopy(global_model))

#     for epoch in range(args.epoch):  # Define num_epochs as needed
#         client_losses = []
#         client_accs = []

#         # Train each client model and evaluate its accuracy
#         for client_id, train_loader in enumerate(federated_train_loaders):
#             optimizer = torch.optim.AdamW(local_models[client_id].parameters(), lr=args.lr, weight_decay=args.weight_decay)
#             local_loss = federated_train_step(args,local_models[client_id],global_model, train_loader, optimizer, criterion,attack, device)
#             client_losses.append(local_loss)

#             # Evaluate the local model
#             local_accuracy = evaluate(local_models[client_id], test_loader, device)
#             client_accs.append(local_accuracy)
#             print(f"Epoch :{epoch+1} client_id:{client_id + 1} client Test Accuracy: {local_accuracy:.2f}%, client Loss: {local_loss:.4f}")

#             # Save per-client results
#             client_results['epoch'].append(epoch + 1)
#             client_results['client_id'].append(client_id + 1)
#             client_results['client_loss'].append(local_loss)
#             client_results['client_accuracy'].append(local_accuracy)

#         # Aggregate weights and calculate global model's performance
#         global_model, local_models = average_weights_v2(args, global_model, local_models, data_sizes)


#         # Evaluate global model for accuracy
#         global_accuracy = global_test_accuracy(args, global_model,local_models, test_loader, device)

#         # Calculate global loss (if needed, you may want to run it through a loss function using a dataset)
#         global_loss = sum(client_losses) / len(client_losses)  # Example averaging client losses for global loss approximation

#         # Calculate the average client accuracy
#         client_avg_accuracy = sum(client_accs) / len(client_accs)

#         global_results['epoch'].append(epoch + 1)
#         global_results['global_loss'].append(global_loss)
#         global_results['global_accuracy'].append(global_accuracy)
#         global_results['client_avg_accuracy'].append(client_avg_accuracy)

#         print(f"Epoch {epoch+1}: Global Test Accuracy: {global_accuracy:.2f}%, Global Loss: {global_loss:.4f}")
        
#     # Save model:
#     save_model(args, global_model, local_models)

#     # Convert results to pandas DataFrames
#     client_df = pd.DataFrame(client_results)
#     global_df = pd.DataFrame(global_results)
    
#     return client_df, global_df


def save_model(args, global_model, clients_model):
    if args.alg.lower() == 'fedbn':
        for i in range(len(clients_model)):
            outpath1 = os.path.join(args.outbase, f'Model_{args.alg}_attack_{args.attack}_client{i}.pth')
            torch.save(clients_model[i].state_dict(), outpath1)
    else:
        outpath1 = os.path.join(args.outbase, f'Model_{args.alg}_attack_{args.attack}.pth')
        torch.save(global_model.state_dict(), outpath1)
    print("The model(s) had been saved")

def global_test_accuracy(args, global_model,local_models, test_loader, device):
    if args.alg.lower() == 'fedbn':
        accuracy_list_client = []
        for client_idx in range(args.num_clients):
            accuracy_list_client.append(evaluate(local_models[client_idx], test_loader, device))
        return  (sum(accuracy_list_client) / len(accuracy_list_client))

    else:
        return evaluate(global_model, test_loader, device)

def average_weights_v2(args, server_model, models, client_weights, local_models):
    with torch.no_grad():
        # aggregate params
        if args.alg.lower() == 'fedbn':
            for key in server_model.state_dict().keys():
                if 'bn' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(len(client_weights)):
                        temp += ( client_weights[client_idx]  / sum(client_weights) ) * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(args.num_clients):
                        local_models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        else:
            for key in server_model.state_dict().keys():
                # num_batches_tracked is a non trainable LongTensor and
                # num_batches_tracked are the same for all clients for the given datasets
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp +=  ( client_weights[client_idx]  / sum(client_weights) ) * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(args.num_clients):
                        local_models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, local_models



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



# def average_weights_v2(args, server_model, models, client_weights):
#     with torch.no_grad():
#         # aggregate params
#         if args.alg.lower() == 'fedbn':
#             for key in server_model.state_dict().keys():
#                 if 'bn' not in key:
#                     temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
#                     for client_idx in range(args.num_clients):
#                         temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
#                     server_model.state_dict()[key].data.copy_(temp)
#                     for client_idx in range(args.num_clients):
#                         models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
#         else:
#             for key in server_model.state_dict().keys():
#                 # num_batches_tracked is a non trainable LongTensor and
#                 # num_batches_tracked are the same for all clients for the given datasets
#                 if 'num_batches_tracked' in key:
#                     server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
#                 else:
#                     temp = torch.zeros_like(server_model.state_dict()[key])
#                     for client_idx in range(len(client_weights)):
#                         temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
#                     server_model.state_dict()[key].data.copy_(temp)
#                     for client_idx in range(len(client_weights)):
#                         models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
#     return server_model, models
