import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
import time
import sys

from preprocessing import DataPreprocessor
from neural_networks import AudioEncoder, IMUEncoder, ProjectionNet, TrainedAudioEncoder, TrainedIMUEncoder
from contrastive_loss import ContrastiveLoss

def generate_pseudo_labels(real_data):
    """
        Since an unsupervised approach is adopted, pseudo labels are generated for real data using K-means clustering.
        Number of clusters = 3 corresponding to the 3 pressure levels.
        Since sklearn's KMeans may give different labels to the same clusters on different runs, label mapping might be required. 
        Use temp_real_labels and cluster_labels to check this and modify the code using : 
        label_mapping = {0: 1, 1: 3, 2: 2}
        cluster_labels = np.vectorize(label_mapping.get)(cluster_labels)        
    """
    temp = real_data
    temp = temp.reshape((len(real_data),-1))
    scaler = StandardScaler()
    temp = scaler.fit_transform(temp)
    
    kmeans = KMeans(n_clusters=3, n_init=100)
    kmeans.fit(temp)
    cluster_labels = kmeans.labels_
    return cluster_labels

def create_batches(X, y, batch_size=100):
    """
        This function creates batches with each batch having same number of samples for each label
    """
    idx = []
    unique_labels = np.unique(y)
    min_lab_count = np.min([len(np.where(y == i)[0]) for i in unique_labels])
    for i in unique_labels:
        idx.extend(np.random.choice(np.where(y == i)[0], size=min_lab_count, replace=False))
    np.random.shuffle(idx)
    X, y = X[idx], y[idx]
    num_batches = len(X)//batch_size
    lab_size = batch_size//len(unique_labels)
    
    for i in range(num_batches):
        indices = []
        for i in unique_labels:
            indices.extend(np.random.choice(np.where(y == i)[0], size=lab_size, replace=False))
        last = [np.random.choice(np.arange(0, len(X)))]
        indices.extend(last)
        np.random.shuffle(indices)
        batch_X, batch_y = X[indices], y[indices]
        yield batch_X, batch_y

def plot_label_histogram(sim_labels, real_labels):
    """
        To check class imbalance
    """
    fig, ax = plt.subplots(1, 2,figsize=(6,2))
    ax[0].hist(sim_labels)
    ax[0].set_title('Simulation labels')
    ax[1].hist(real_labels)
    ax[1].set_title('Real labels')
    plt.show()

# Training code + Data Generation for WGP
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Pass proper paths for simulated and real data folders")
        sys.exit(1)
    
    path_to_simulated_data = sys.argv[1]
    path_to_real_data = sys.argv[2]

    # Preprocess the raw data
    sim = DataPreprocessor(path_to_simulated_data)
    real = DataPreprocessor(path_to_real_data)
    sim_data, sim_labels = sim.preprocessing_pipeline()
    real_data, temp_real_labels = real.preprocessing_pipeline()

    # Get pseudo labels
    real_labels = generate_pseudo_labels(real_data)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Split preprocessed data into audio and IMU data (data -> 1 x audio amplitude, 6 x imu features)
    sim_audio_data, sim_imu_data = sim_data[:,:,:1], sim_data[:,:,1:]    
    real_audio_data, real_imu_data = real_data[:,:,:1], real_data[:,:,1:]
    
    # Create batches for parallel computing
    sim_audio_batch = list(create_batches(sim_audio_data,sim_labels))
    sim_imu_batch = list(create_batches(sim_imu_data,sim_labels))
    real_audio_batch = list(create_batches(real_audio_data,real_labels))
    real_imu_batch = list(create_batches(real_imu_data,real_labels))
    
    # Initialize the models
    audio_model = AudioEncoder()
    audio_model.to(device)
    
    imu_model = NewIMUEncoder()
    imu_model.to(device)
    
    prj_net = ProjectionNet(audio_model, imu_model)
    prj_net.to(device)
    
    # Contrastive Loss Object
    loss = ContrastiveLoss(0.5, audio_model, imu_model, prj_net)
    optimizer = optim.Adam(list(audio_model.parameters())+list(imu_model.parameters())+list(prj_net.parameters()), lr=0.0001)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=4)
    
    # TRAINING - CONTRASTIVE LEARNING
    # Training Start ---------------------------------------------------------------------------------
    num_epochs = 10
    num_train_batches = len(sim_audio_batch)
    
    # Set models to train
    audio_model.train()
    imu_model.train()
    prj_net.train()
    
    for epoch in range(num_epochs):
        start_time = time.time()
        loss_threshold = 1e-4
        prev_loss = float('inf')
        loss_not_improving_count = 0
        max_not_improving_count = 3
        
        for b in range(num_train_batches):
            # Get a batch
            # sim_audio_labels = sim_imu_labels for a batch, same for real data 
            sim_audio_dataset, sim_audio_labels = torch.from_numpy(sim_audio_batch[b][0]).float(), torch.from_numpy(sim_audio_batch[b][1]).long()
            sim_imu_dataset, _ = torch.from_numpy(sim_imu_batch[b][0]).float(), torch.from_numpy(sim_imu_batch[b][1]).long()
            real_audio_dataset, real_audio_labels = torch.from_numpy(real_audio_batch[b][0]).float(), torch.from_numpy(real_audio_batch[b][1]).long()
            real_imu_dataset, _ = torch.from_numpy(real_imu_batch[b][0]).float(), torch.from_numpy(real_imu_batch[b][1]).long()

            # Pass simulated data batch to contrastive to form positive and negative pairs
            loss.get_simulated_data(sim_audio_labels, sim_audio_dataset, sim_imu_dataset)
            
            bl = 0.0 # batch loss

            # For each target anchor in real data, train the encoders and projection netowrk using contrastive loss
            for i in range(len(real_audio_labels)):

                # Get the output representations from Encoders
                z1_, z2_ = audio_model(real_audio_dataset[i].to(device)), imu_model(real_imu_dataset[i].to(device))
                lab = real_audio_labels[i] 
                lab = lab.to(device)
                z1_, z2_ = z1_.to(device), z2_.to(device)

                # Pass the encoder representations to projection network
                r1_, r2_ = prj_net(z1_,z2_)

                # Concatenate the projection network representations and flatten them
                r1_r2_ = torch.cat((r1_, r2_), dim=1)
                r1_r2_ = r1_r2_.view(r1_r2_.shape[0]*r1_r2_.shape[1],-1)
                r1_r2_ = r1_r2_.to(device)

                optimizer.zero_grad()

                # Pass the flattened reps and pseudo label to contrastive loss
                l = loss(r1_r2_, lab)
                
                l.backward()
                optimizer.step()
                
                bl = bl + l.item()
                
            average_batch_loss = bl/len(real_audio_labels)
            print(f'Batch {b}, loss = {average_batch_loss}')
            scheduler.step()

            # Early stopping
            loss_change = np.abs(prev_loss - average_batch_loss)
            if loss_change < loss_threshold:
                loss_not_improving_count += 1
            else:
                loss_not_improving_count = 0
        
            prev_loss = average_batch_loss
    
            if loss_not_improving_count >= max_not_improving_count:
                print(f"Stopping training early at epoch {epoch+1}")
                break

            # Save checkpoints
            checkpoint_path = f'models/checkpoints/checkpoint_batch{b+1}.pth'
            torch.save({
                'epoch': epoch+1,
                'batch': b+1,
                'audio_model_state_dict': audio_model.state_dict(),
                'imu_model_state_dict': imu_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'prj_net_state_dict': prj_net.state_dict(),
                'loss': average_batch_loss,
            }, checkpoint_path)

        epoch_loss = prev_loss
        end_time = time.time()
        epoch_time = end_time - start_time
        print("Epoch training time:", epoch_time)
        print(f"Epoch [{epoch+1}/{num_epochs}]  Training Loss: {epoch_loss} ------------------------")

    # Training End ---------------------------------------------------------------------------------

    # Save the Weights of Domain Invariant Encoders
    audio_param = audio_model.state_dict()
    torch.save(audio_param, 'models/audio_param32x20.pth')
    imu_param = imu_model.state_dict()
    torch.save(imu_param, 'models/imu_param32x20.pth')

    # Generate data for Warped Gaussian Process using Encoders only. The projection network is dropped. Only Real data is used

    # Load saved weights into the trained encoders
    audio_model = TrainedAudioEncoder()
    audio_model.to(device)
    imu_model = TrainedIMUEncoder()
    imu_model.to(device)

    audio_parameters = torch.load('models/audio_param32x20.pth')
    audio_model.load_state_dict(audio_parameters)
    imu_parameters = torch.load('models/imu_param32x20.pth')
    imu_model.load_state_dict(imu_parameters) 

    # Freeze the layers
    for param in audio_model.parameters():
        param.requires_grad = False
    
    for param in imu_model.parameters():
        param.requires_grad = False

    # Training data for WGP
    b = np.random.randint(0, num_train_batches)
    real_audio_dataset, _ = torch.from_numpy(real_audio_batch[b][0]).float(), torch.from_numpy(real_audio_batch[b][1]).long()
    real_imu_dataset, real_imu_labels = torch.from_numpy(real_imu_batch[b][0]).float(), torch.from_numpy(real_imu_batch[b][1]).long()

    # here 6 corresponds to the 6 features in IMU reps and 6 duplications of audio reps. 32 is the number of output channels of encoders
    audio_rep, imu_rep = torch.empty(0,6,32), torch.empty(0,6,32)
    audio_rep, imu_rep = audio_rep.to(device), imu_rep.to(device)
    
    with torch.no_grad():
        for i in range(len(real_imu_labels)):
            au, im = real_audio_dataset[i], real_imu_dataset[i]
            au, im = au.to(device), im.to(device)
            out1, out2 = audio_model(au), imu_model(im)
            # average over time-steps
            out1, out2 = torch.mean(out1, dim=2), torch.mean(out2, dim=2)
            out1, out2 = out1.unsqueeze(0), out2.unsqueeze(0)
            audio_rep = torch.cat([audio_rep, out1], dim=0)
            imu_rep = torch.cat([imu_rep, out2], dim=0)
    
        torch.save(audio_rep,'models/audio_rep6x32.pt')
        torch.save(imu_rep, 'models/imu_rep6x32.pt')
        torch.save(real_imu_labels,'models/clf_train_labs.pt') # these labels will be used to train classifier

    # Testing data for WGP
    bt = next(i for i in np.random.permutation(num_train_batches) if i != b)
    real_audio_dataset, _ = torch.from_numpy(real_audio_batch[bt][0]).float(), torch.from_numpy(real_audio_batch[bt][1]).long()
    real_imu_dataset, real_imu_labels = torch.from_numpy(real_imu_batch[bt][0]).float(), torch.from_numpy(real_imu_batch[bt][1]).long()
    
    test_audio_rep, test_imu_rep = torch.empty(0,6,32), torch.empty(0,6,32)
    test_audio_rep, test_imu_rep = test_audio_rep.to(device), test_imu_rep.to(device)
    
    with torch.no_grad():
        for i in range(len(test_imu_labels)):
            au, im = test_audio_dataset[i], test_imu_dataset[i]
            au, im = au.to(device), im.to(device)
            out1, out2 = audio_model(au), imu_model(im)
            # average over time-steps
            out1, out2 = out1.mean(axis=2), out2.mean(axis=2)
            out1, out2 = out1.unsqueeze(0), out2.unsqueeze(0)
            test_audio_rep = torch.cat([test_audio_rep, out1], dim=0)
            test_imu_rep = torch.cat([test_imu_rep, out2], dim=0)
    
        torch.save(test_audio_rep,'models/test_audio_rep6x32.pt')
        torch.save(test_imu_rep, 'models/test_imu_rep6x32.pt')
        torch.save(test_imu_labels, 'models/clf_test_labs.pt') # labels to test classifier

    # x----------------------------x------------------------------x-----------------------------x---------------------------x