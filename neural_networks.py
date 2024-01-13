import torch
import torch.nn as nn

class AudioEncoder(nn.Module):
        """
            Conv1d + LSTM is used for time-series data
            
            Input audio data shape = (batch_size x sequence_length x num_channels) 
            sequence_length = downsampled_time_steps = 20
            num_channels = 1 (amplitude only) = num_input_channels in Conv1d layer

            Conv1d input shape = (batch_size x num_channels x sequence_length) 
            LSTM input shape = (batch_size x sequence_length x num_channels) 
            Hidden size for LSTM = 32
            
            https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        """
    def __init__(self):
        super(AudioEncoder, self).__init__()
        # Conv1d for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size = 1), # inp_channels = 1
            nn.PReLU(),
            nn.Conv1d(32, 64, kernel_size = 1),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(64, 128, kernel_size = 1),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(128, 256, kernel_size = 1),
            nn.PReLU(),
        )
        # LSTM for sequential modeling
        self.lstm = nn.LSTM(input_size=256, hidden_size=32, num_layers=2, batch_first=True, dropout=0.2) # hidden_size = 32

    def forward(self, x):
        x = x.unsqueeze(0).permute(0, 2, 1)
        x = self.conv_layers(x)
        x = x.permute(0,2,1)
        x, _ = self.lstm(x)
        x = x.permute(0,2,1)
        x = x.squeeze(0)
        return x
        
class IMUEncoder(nn.Module):
        """
            Input imu data shape = (batch_size x sequence_length x num_channels) 
            sequence_length = downsampled_time_steps = 20
            num_imu_features = 6 (ACC X, ACC Y, ACC Z, GYRO X, GYRO Y, GYRO Y)
            
               ┌─────────────────────┐
               │ Input (6 features)  │
               └─────────────────────┘
                   │  │  │  │  │  │    Pass each feature separately into CNN + LSTM block for parallel processing.
                   ▼  ▼  ▼  ▼  ▼  ▼
               ┌─────────────────────┐
               │ Conv1D + LSTM block │
               └─────────────────────┘
                   │  │  │  │  │  │    Parallel processing allows the model to capture unique patterns and dependencies specific to each feature.
                   ▼  ▼  ▼  ▼  ▼  ▼
               ┌─────────────────────┐
               │       Average       │
               └─────────────────────┘
                          │            Average for a comprehensive representation, thus enhancing information extraction from the IMU data.
                          ▼
                      ┌────────┐
                      │ Output │
                      └────────┘
        """
    def __init__(self):
        super(IMUEncoder, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size = 1),
            nn.PReLU(),
            nn.Conv1d(32, 64, kernel_size = 1),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(64, 128, kernel_size = 1),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(128, 256, kernel_size = 1),
            nn.PReLU(),
        )
        self.lstm = nn.LSTM(input_size=256, hidden_size=32, num_layers=2, batch_first=True, dropout=0.2)

    def forward(self, x):
        output = []
        for channel in range(x.shape[1]):
            ch_data = torch.select(x,1,channel)
            ch_data = ch_data.unsqueeze(0)
            ch_data = ch_data.unsqueeze(0)
            ch_out = self.block(ch_data)
            ch_out = ch_out.permute(0,2,1)
            ch_out, (_,_) = self.lstm(ch_out)
            ch_out = ch_out.squeeze(0).permute(1,0)
            output.append(ch_out)
        stack = torch.stack(output, dim=0)
        avg_tensor = torch.mean(stack, dim=0,keepdim=True)
        avg_tensor = avg_tensor.squeeze(0)
        return avg_tensor

class ProjectionNet(nn.Module):
    """
        Projection Network maps the unimodal features of different encoders to the same dimension
    """
    def __init__(self, encoder1, encoder2):
        super(ProjectionNet, self).__init__()
        self.encoder1 = encoder1 # audio encoder
        self.encoder2 = encoder2 # imu encoder

        self.layers1 = nn.Sequential(
            nn.Conv1d(32, 128, kernel_size = 1),
            nn.PReLU(),
            nn.Conv1d(128,128, kernel_size = 1),
            nn.PReLU()
        )
        self.layers2 = nn.Sequential(
            nn.Conv1d(32, 128, kernel_size = 1), 
            nn.PReLU(),
            nn.Conv1d(128,128, kernel_size = 1),
            nn.PReLU()
        )
    def forward(self, x1, x2):
        out1 = self.layers1(x1)
        out2 = self.layers2(x2)
        out1, out2 = out1.squeeze(0), out2.squeeze(0)
        return out1, out2

"""
    After training the encoders using Contrastive Learning, they achieve Domain Invariance. 
    The representations learned by these encoders are then passed into the WGP(WGP) models.

    Since each feature of the IMU was learned independently, the final output is adjusted to preserve the 6 distinct features instead of averaging.
    No re-training is necessary as this modification involves no changes to the model weights.

    As WGP requires audio and IMU representations to have same dimensions, the audio representation is duplicated 6 times.
"""

class TrainedAudioEncoder(AudioEncoder):
    def __init__(self):
        super(TrainedAudioEncoder, self).__init__()

    def forward(self, x):
        x = x.unsqueeze(0).permute(0, 2, 1)
        x = self.conv_layers(x)
        x = x.permute(0,2,1)
        x, _ = self.lstm(x)
        x = x.permute(0,2,1)
        x = x.repeat(6, 1, 1)
        return x

class TrainedIMUEncoder(IMUEncoder):
    def __init__(self):
        super(TrainedIMUEncoder, self).__init__()

    def forward(self, x):
        output = []
        for channel in range(x.shape[1]):
            ch_data = torch.select(x,1,channel)
            ch_data = ch_data.unsqueeze(0)
            ch_data = ch_data.unsqueeze(0)
            ch_out = self.block(ch_data)
            ch_out = ch_out.permute(0,2,1)
            ch_out, (_,_) = self.lstm(ch_out)
            ch_out = ch_out.squeeze(0).permute(1,0)
            output.append(ch_out)
        stack = torch.stack(output, dim=0)
        return stack

num_classes = 3

class Classifier(nn.Module):
    """
        Classifier that takes representations from Multi-task Warped Gaussian Process as input and gives pressure level as output
    """
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(64, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = x.view(-1)  # Flatten
        x = self.layers(x)
        return x