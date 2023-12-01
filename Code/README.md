# A Dynamic Domain Adaptation Deep Learning Network for EEG-based Motor Imagery Classification
We provide a Dynamic Domain Adaptation Based Deep Learning Network (DADLNet) for addressing the inter-subject and inter-session variability in MI-BCI. We replace traditional EEG with 3D array and use 3D convolution to learn temporal and spatial features. For the purpose of our model can better capture spatial-temporal information, we add an attention method that spatially combines convolutional channels. Furthermore, we develop a dynamic domain adaptation (DDA) strategy to adapt to the source domain in different scenarios, utilize maximum mean discrepancy (MMD) loss functionto reduce the distance between the source and target domains to achieve the best results. We verify the performance of the proposed method on BCI competition IV 2a and OpenBMI datasets. Under the intra-subject experiment, the accuracy rates of 70.42±12.44 and 73.91±11.28 were achieved on the OpenBMI and BCIC IV 2a datasets.<br><br>
**The following two folders store the experimental code and experimental results respectively. The detailed explanation is as follows：**
## Code
This folder stores the data preprocessing, training, and testing codes of DADLNet.<br>
Instructions to Run DADLNet on OpenBMI and BCI competition IV-2a dataset. <br>
### Step 1: Set-up a virtual environment
1. Create a new virtual environment with python3.8: DADLNet<br>
```conda env create -n DADLNet python=3.8```
2. Load the virtualenv and install required packages<br>
```conda activate DADLNet```<br>
### Step 2: Get the data
#### BCI competition IV-2a：
The BCI competition IV-2a dataset is only accessible to registered users. So, you need to manually download the data.
1. Register and download the train and test data at http://www.bbci.de/competition/iv/  or http://bbci.de/competition/iv/download .
2. Download the test data labels from http://bbci.de/competition/iv/results/ds2a/true_labels.zip
``` 
  -A01T.mat
  -A01T.gdf
  .
  .
  .

  -A01E.mat
  -A01E.gdf
  .
  .
  . 
```
#### OpenBMI:
OpenBMI dataset is accessible by FTP protocol. <br>
Download the train and test data as ftp://parrot.genomics.cn/gigadb/pub/10.5524/100001_101000/100542/DataSet/OpenBMI/raw
``` 
  -sess01_subj01_EEG_MI.mat
  -sess01_subj02_EEG_MI.mat
  .
  .
  .

  -sess02_subj01_EEG_MI.mat
  -sess02_subj02_EEG_MI.mat
  .
  .
  . 
```
#### Phase1:Pre-train
#####  Step 1: Run the data process codes
First configure the data path and the corresponding data format in the yml file, and then execute the prep_time_domain.py file to generate the training data.<br>
```python prep_time_domain.py ./MI/model/1S_FB1-100_S100_chanSeq_baselinSED_Downsampl.yml```
#####  Step 2: Run the train codes
```cd ./MI/TrainSub```
```python train.py ../model/1S_FB1-100_S100_chanSeq_baselinSED_Downsampl.yml```
#####  Step 3: Run the test codes
```cd ./MI/TrainSub```
```python test.py ../model/1S_FB1-100_S100_chanSeq_baselinSED_Downsampl.yml```
#### Phase2:Domain adaptation
```cd ./MI/DA```
intra_sub_BCIC_IV_2A：<br>
```python BCIC_IV_2A_intra_sub.py ./1S_FB1-100_S100_chanSeq_DA.yml```
intra_sub_OpenBMI:<br>
```python OpenBMI_intra_sub.py ./1S_FB1-100_S100_chanSeq_DA.yml```
inter_sub_BCIC_IV_2A：<br>
```python BCIC_IV_2A_inter_sub.py ../model/1S_FB1-100_S100_chanSeq_DA.yml```

The code for the comparison experiment is shown in the link below:
* EEGNet：https://github.com/aliasvishnu/EEGNet
* EEG-adapt：https://github.com/zhangks98/eeg-adapt
* FBMSNet：https://github.com/Want2Vanish/FBMSNet
* MIN2Net：https://github.com/MIN2Net/MIN2Net.github.io
## Experimental results
This folder stores the specific experimental results and model files for the two datasets of BCIC IV 2A and OpenBMI under the five models(DADLNet,EEGNet,EEG-adapt,FBMSNet,MIN2Net). Different groupings are set under each dataset file to distinguish between Intra-subject and Inter-subject.
