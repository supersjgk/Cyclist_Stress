import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    Cross-domain Contrastive Loss (CDCL) Class with the Simulated data being the source domain and Real data being the target domain. 
    Positive pairs: 2 samples with same labels in differnt domains
    Negative pairs: 2 samples with different labels in different domains

    *******************Check out the CDCL.md file for formula.**********************
    
    Hyperparameters: Temperature - controls how much to penalize the negative pairs
"""

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature, audio_model, imu_model, prj_net):
        super(ContrastiveLoss, self).__init__()
        self.T = temperature 
        self.audio_model = audio_model
        self.imu_model = imu_model
        self.prj_net = prj_net

    def get_simulated_data(self, sim_labels, sim_audio_dataset, sim_imu_dataset):
        # Gets one batch form source domain data to form positive and negative pairs with a target anchor from a target batch.
        self.sim_labels = sim_labels
        self.sim_audio_dataset = sim_audio_dataset
        self.sim_imu_dataset = sim_imu_dataset

    def forward(self, r1_r2_, lab):
        # Target anchor: The output representations from Audio and IMU are concatenated and flattened
        Zti = r1_r2_
        
        # pseudo label of target anchor
        Yti = lab
        
        # indices of samples from source domain that form a positive pair with target anchor
        ind = [i for i in range(len(self.sim_labels)) if self.sim_labels[i]==Yti] 

        # number of positive pairs 
        Ps = len(ind)
        
        den = torch.zeros(1)
        den = den.to(lab.device)
        log_sum = torch.zeros(1)
        log_sum = log_sum.to(lab.device)

        # calculating the denominator
        for i in range(len(sim_labels)):
            # for each sample in source domain, get the encoder representations, concatenate and flatten them
            z1, z2 = self.audio_model(self.sim_audio_dataset[i].to(lab.device)), self.imu_model(self.sim_imu_dataset[i].to(lab.device))
            r1, r2 = self.prj_net(z1, z2)
            r1r2 = torch.cat((r1, r2), dim=1)
            r1r2 = r1r2.view(r1r2.shape[0]*r1r2.shape[1],-1)
            if r1r2.device != lab.device:
                r2r2 = r1r2.to(lab.device)
            Zsj = r1r2
            # denominator will remain same for each batch
            den += torch.exp((F.cosine_similarity(Zti, Zsj, dim=0))*(1/self.T))

        for i in ind:
            # for each positive pair, get the encoder reps, concatenate and flatten them
            z1, z2 = self.audio_model(self.sim_audio_dataset[i].to(lab.device)), self.imu_model(self.sim_imu_dataset[i].to(lab.device))
            r1, r2 = self.prj_net(z1,z2)
            r1r2 = torch.cat((r1, r2), dim=1)
            r1r2 = r1r2.view(r1r2.shape[0]*r1r2.shape[1],-1)
            if r1r2.device != lab.device:
                r2r2 = r1r2.to(lab.device)
            Zsp = r1r2
            
            # calculating the numerator
            num = torch.exp((F.cosine_similarity(Zti, Zsp, dim=0))*(1/self.T))
            
            # calculate the log sum
            log_sum += torch.log(torch.div(num,den))

        # Final loss value calculated using just one target anchor from a batch
        Lcdc_ti = (-1/(Ps*len(self.sim_labels)))*log_sum
        
        return Lcdc_ti