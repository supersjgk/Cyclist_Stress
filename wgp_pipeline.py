import torch
import torch.nn as nn
import gpytorch
import numpy as np
import time

from warped_gaussian_process import WGP, WarpClass

"""
    NOTE:
    One batch of representations from trained encoders has a shape (100, 6, 32, 20), 
    representing (num_samples, num_imu_features, num_output_channels, num_time_steps). 
    We already averaged over time-steps in the uda_contrastive_loss_pipeline file to capture the overall trend while generating data for WGP.
    Now, our data has a shape (100, 6, 32). Each of the 32 output channels from CNN+LSTM encoders learns a different feature so we treat them as independent. 
    Now, each channel has data of shape (100, 6). Instead of passing 6 as the number of tasks in the Multi-task Warped Gaussian Process (MTWGP), 
    we pass 100 as the number of tasks. This approach allows the model to learn the unique characteristics of each instance, making it more robust.
"""

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Loading representations from domain invariant encoders
    _train_x, _train_y = torch.load('models/imu_rep6x32.pt'), torch.load('models/audio_rep6x32.pt')
    _test_x, _test_y = torch.load('models/test_imu_rep6x32.pt'), torch.load('models/test_audio_rep6x32.pt')

    # TRAINING Start ----------------------------------------------------------------------------------------
    train_x, train_y = _train_x[:100].permute(2,1,0), _train_y[:100].permute(2,1,0)
    train_x, train_y = train_x.to(device), train_y.to(device)

    num_channels = 32
    models = []
    warped_models = []
    torch.autograd.set_detect_anomaly(True)

    # We'll have only one set of trainable parameters for warping function across all channels and all tasks.
    warped_model = WarpClass()
    warped_model = warped_model.to(device)
    
    start = time.time()
    
    for c in range(num_channels):

        # Define a MTWGP for a channel data
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=100)
        likelihood = likelihood.to(device)

        # Apply the warping function
        z = warped_model(train_y[c])

        # Define the GP model over z rather than y
        model = WGP(train_x[c],z,likelihood)
        model = model.to(device)
    
        model.train()
        likelihood.train()
        warped_model.train()

        # Train the parameters of GPyTorch model and Warped class together.
        optimizer = torch.optim.Adam(list(model.parameters())+list(warped_model.parameters()), lr=0.1)

        # Negative Log likelihood loss
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        # Number of optimization steps to update the parameters and minimize the objective function
        training_iterations = 20
        
        print('Channel:',c+1,'-'*45)
    
        optimizer.zero_grad()
        for i in range(training_iterations):
            optimizer.zero_grad()

            # Forward pass
            dist = model(train_x[c])

            # Negative log likelihood loss term associated just with the Multi-task Gaussian Process
            nll = -mll(dist,z)

            # Additional loss term associated with the Warping function
            loss_term = warped_model.loss(train_y[c])

            # Combined loss
            loss = nll - loss_term
    
            loss.backward(retain_graph=True)
            optimizer.step()

            # Clamp the paramters to make sure they are positive
            warped_model.a.data = torch.clamp(warped_model.a.data, min=1e-2)
            warped_model.b.data = torch.clamp(warped_model.b.data, min=0.49)
            warped_model.c.data = torch.clamp(warped_model.c.data, min=1e-2)
            if i==19:
                print(f'a:{warped_model.a.item()},b:{warped_model.b.item()},c:{warped_model.c.item()},d:{warped_model.d.item()}')
                print('Loss:', loss.item())

        # After training, we would have a separate set of MTWGP parameters for each channel but only one set of parameters for Warping function
        model_path = f'models/wgp/wgp_{c}.pth'
        # Save parameters of each independent channel
        torch.save(model.state_dict(), model_path)
        models.append(model_path)
    
        print("\n")
    
    end = time.time()
    print('Total Time taken:', end-start)

    # TRAINING End ----------------------------------------------------------------------------------------

    # Save Warping function parameters
    torch.save(warped_model.state_dict(),f'models/warped_model.pth')

    # Write model paths in a file
    with open("models/model_paths.txt", "w") as f:
        for path in models[-32:]:
            f.write(path + "\n")

    print("TRAINING COMPLETE AND MODELS SAVED")

    # TESTING Start ----------------------------------------------------------------------------------------

    with open("models/model_paths.txt", "r") as f:
        model_path = f.read().splitlines()

    # Load trained model
    warped_model = WarpClass()
    warped_model.load_state_dict(torch.load('models/warped_model.pth'))
    warped_model = warped_model.to(device)

    # Load testing data
    # clf_test_x and clf_test_y will be used for testing final classifier
    clf_test_x = _test_x
    clf_test_y = torch.zeros((0,6,32))
    clf_test_y = clf_test_y.to(device)

    test_x, test_y = _test_x.permute(2,1,0), _test_y.permute(2,1,0)
    output_rep = torch.zeros((0,test_x.shape[1],test_x.shape[2]))
    output_rep = output_rep.to(device)

    # Load parameters corresponding to each channel separately and perform testing
    for c in range(num_channels):
        
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=100)
        likelihood = likelihood.to(device)
        pt = model_path[c]
    
        model_state = torch.load(pt)
        model = WGP(likelihood = likelihood)
        model.load_state_dict(model_state)
        model = model.to(device)
        
        model.eval()
        likelihood.eval()
        warped_model.eval()
    
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = likelihood(model(test_x[c]))#---------------------
            mean = predictions.mean
            variance = predictions.variance
            
            # Applying inverse of warped function to get final output
            out = warped_model.inverse(mean)
            
            output_rep = torch.cat([output_rep, out.unsqueeze(0)], dim=0)
    clf_test_y = torch.cat([clf_test_y, output_rep.permute(2,1,0)],dim=0)

    # TESTING End ----------------------------------------------------------------------------------------

    # Calculating Mean Squared Error between Audio data and the representations obtained from MTWGP by passing IMU data
    mse = torch.mean(torch.square(_test_y - clf_test_y))
    print("TESTING COMPLETE. Mean Squared Error:", mse)

    # Save Classifier testing data
    torch.save(clf_test_x,'clf_test_x.pt')
    torch.save(clf_test_y,'clf_test_y.pt')

    # Generating training data for classifier - everything is same as testing above
    warped_model = WarpClass()
    warped_model.load_state_dict(torch.load('models/warped_model.pth'))
    warped_model = warped_model.to(device)

    # We use the original training data, pass it into trained MTWGP and get classifier training data
    clf_train_x = _train_x
    clf_train_y = torch.zeros((0,6,32))
    clf_train_y = clf_train_y.to(device)
    
    train_x, train_y = _train_x.permute(2,1,0), _train_y.permute(2,1,0)
    output_rep = torch.zeros((0,train_x.shape[1],train_x.shape[2]))
    output_rep = output_rep.to(device)
    
    for c in range(num_channels):
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=100)
        likelihood = likelihood.to(device)
        pt = model_path[c]
    
        model_state = torch.load(pt)
        model = WGP(likelihood = likelihood)
        model.load_state_dict(model_state)
        model = model.to(device)
        
        model.eval()
        likelihood.eval()
        warped_model.eval()
    
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = likelihood(model(train_x[c]))
            mean = predictions.mean
            variance = predictions.variance
            out = warped_model.inverse(mean)
            output_rep = torch.cat([output_rep, out.unsqueeze(0)], dim=0)
    clf_train_y = torch.cat([clf_train_y, output_rep.permute(2,1,0)],dim=0)

    # Saving classifier training data
    torch.save(clf_train_x,'clf_train_x.pt')
    torch.save(clf_train_y,'clf_train_y.pt')    
