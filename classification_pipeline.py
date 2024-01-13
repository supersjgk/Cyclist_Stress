import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import time

from neural_networks import Classifier

"""
    Representations obtained from MTWGP have shape (100,6,32) representing (num_samples, num_imu_features, num_output_channels). 
    We average over the feature dimension to capture overall pattern and pass into the classifier.
    The output is pressure level (1 - low, 2 - medium, 3 - high)
"""

if __name__ == "__main__":
    # Load data
    clf_train_x = torch.load('clf_train_x.pt')
    clf_train_y = torch.load('clf_train_y.pt')
    train_labs = torch.load('models/clf_train_labs.pt')
    clf_test_x = torch.load('clf_test_x.pt')
    clf_test_y = torch.load('clf_test_y.pt')
    test_labs = torch.load('models/clf_test_labs.pt')

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Average over features
    clf_train_x = torch.mean(clf_train_x, dim=1)
    clf_train_y = torch.mean(clf_train_y, dim=1)
    clf_test_x = torch.mean(clf_test_x, dim=1)
    clf_test_y = torch.mean(clf_test_y, dim=1)

    num_classes = 3
    
    # Initialize the objects
    classifier = Classifier(num_classes)
    classifier.to(device)
    c_loss = nn.CrossEntropyLoss()
    c_loss.to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=0.00001)

    # TRAINING START ----------------------------------------------------------------------
    num_epochs = 1000
    
    classifier.train()
    start_time = time.time()
    
    #x->IMU rep from encoder, y->Audio rep obtained from MTWGP
    for epoch in range(num_epochs):
        for i in range(len(train_labs)):
            t1, t2 = clf_train_y[i], clf_train_x[i]
            t1, t2 = t1.view(t1.shape[0],1), t2.view(t2.shape[0],1)
            lab = train_labs[i]
            lab = lab.to(device)

            # concatenate and flatten the data and pass to classifier
            t1t2 = torch.cat((t1, t2), dim=1)
            t1t2 = t1t2.view(t1t2.shape[0]*t1t2.shape[1],1)
            t1t2 = t1t2.to(device)
            optimizer.zero_grad()
            output = classifier(t1t2)
            output = output.to(device)

            # CrossEntropyLoss expects labels to start from 0 whereas our labels start from 1
            lab = lab-1
            l = c_loss(output, lab)
            
            l.backward()
            optimizer.step()
    
        print(f"Epoch [{epoch+1}/{num_epochs}] classifier Loss: {l.item()}")
    
    end_time = time.time()
    epoch_time = end_time - start_time
    print("TRAINING COMPLETE, Time taken:", epoch_time)

    # TRAINING End ----------------------------------------------------------------------

    # Saving the classifier weights
    clf_param = classifier.state_dict()
    torch.save(clf_param, 'models/clf_64.pth')

    # TESTING Start ----------------------------------------------------------------------
    # Load saved weights
    classifier = Classifier(num_classes)
    clf_parameters = torch.load('models/clf_64.pth')
    classifier.load_state_dict(clf_parameters)

    # Freeze layers
    for param in classifier.parameters():
        param.requires_grad = False

    correct = 0
    total = 0
    pred_list = []
    
    classifier.eval()
    with torch.no_grad():
        for i in range(len(clf_test_x)):
            t1, t2 = clf_test_y[i], clf_test_x[i]
            t1, t2 = t1.view(t1.shape[0],1), t2.view(t2.shape[0],1)
            lab = test_labs[i]
            lab = lab.to(device)
            # concatenate and flatten test data
            t1t2 = torch.cat((t1, t2), dim=1)
            t1t2 = t1t2.view(t1t2.shape[0]*t1t2.shape[1],1)
            t1t2 = t1t2.to(device)
            output = classifier(t1t2)
            output = output.to(device)
            output_probs = F.softmax(output,dim=0)

            # predicted label is the index with max value
            pred = torch.argmax(output_probs).item()
            pred_list.append(pred+1)
            total += 1
            correct += torch.sum(pred == (lab-1)).item()
    
    accuracy = 100 * correct / total
    print("TESTING COMPLETE")
    print(f"Val Accuracy: {accuracy:.2f}%")
    
    # TESTING End ----------------------------------------------------------------------
    