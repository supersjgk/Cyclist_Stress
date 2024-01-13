# Implementation of Cyclist Stress Research Paper

## Code Files
* preprocessing.py - Data Preprocessing, Denoising, Downsampling
* neural_networks.py - Neural networks involved in the project
* contrastive_loss.py - Implementation of Cross Domain Contrastive Loss (CDCL) in PyTorch
* uda_contrastive_loss_pipeline.py - Training the encoders to achieve domain invariance using Contrastive loss
* warped_gaussian_process.py - Implementation of Multi-task Warped Gaussian Process using GPyTorch
* wgp_pipeline.py - Training and Testing MTWGP
* classification_pipeline.py - Training and Testing the final classifier
* CDCL.md - CDCL formula

## Setup
Main Libraries used: [PyTorch](https://github.com/pytorch/pytorch) | [GPyTorch](https://github.com/cornellius-gp/gpytorch) | [SciPy](https://github.com/scipy/scipy) | [noisereduce](https://github.com/timsainb/noisereduce/tree/master)
```
pip install -r requirements.txt
```
Latest versions are recommended

## Modules
### 1. Unsupervised Domain Adaptation with Cross Domain Contrastive Loss
This module uses a multi-modal neural network with separate encoders for Audio and IMU data. The goal is to make the encoder representations invariant across labeled source domain (simulated data) and unlabeled target domain (real-world data). This is achieved by using Cross-domain Contrastive Loss. The encoder representations are first concatenated and flattened, then passed into a projection network to ensure a common dimension space. The loss function encourages the encoders to produce similar representations for semantically related instances from different domains. Once, the training is complete, the projection network is discarded.

For Audio data, amplitude is used, while for IMU, acceleration and gyroscope data along three axes is used. The novelty lies in the parallel branching of each IMU component, enabling the model to learn and capture nuanced patterns independently for enhanced feature extraction.
   
   To run this, first you should have the following data hierarchies:
   
        /Source_Data
        ├── Participant 1
        |   ├── Session 1
        │   │   ├── bike_data.csv - IMU data
        |   |   ├── bike_audio_timestamp.csv - Labels
        |   |   └── bike_audio_timestamp.pcm - Audio data
        |   ├── Session 2 .....
        |   └── Session 3 ..... so on   
        |   
        ├── Participant 2 ..... so on
        |

        /Target_Data
        ├── Participant 1
        |   ├── Session 1
        │   │   ├── bike_data.csv - IMU data
        |   |   └── bike_audio_timestamp.pcm - Audio data
        |   ├── Session 2 .....
        |   └── Session 3 ..... so on   
        |   
        ├── Participant 2 ..... so on
        |
   Then, run the pipeline using:
   ```
   python uda_contrastive_loss_pipeline.py "path/to/source/data" "path/to/target/data"
   ```
   This will run the preprocessing pipeline, create pseudo labels, run the domain adaptation training pipeline, and generate data for MTWGP pipeline.

### 2. Multi Task Warped Gaussian Process (MTWGP)
The representations obtained from domain invariant encoders are used to learn the relationship between IMU data (x) and Audio data (y) using MTWGP. The goal is to minimize energy consumption. Once, the MTWGP is trained, the IMU data can be used to generate the audio representations directly without having to collect audio data through recording. Since, audio data might not be Gaussian, a Warping function is introduced.
   ```
   python wgp_pipeline.py
   ```
   This will run the MTWGP pipeline and generate data for classification pipeline

### 3. Classification
Now, we can use IMU representation obtained from domain invariant encoder, generate audio reprentations using MTWGP, concatenate and flatten them again to train and test the classifier and get the pressure levels (1 - low, 2 - medium, 3 - high).
   ```
   python classification_pipeline.py
   ```
   This will run the final classification and output the pressure levels.

### 4. After training workflow
Record IMU data -> Pass to IMU encoder -> Pass the reps to MTWGP to get Audio reps -> concatenate + flatten -> Pass  to classifier -> Pressure level 
   
