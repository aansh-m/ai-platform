# Semantic Segmentation
Model for semantic segmentation of nuclei from cell images. The model used is U-Net.
### Dataset
The training dataset can be downloaded from the following link:
<br>
https://www.kaggle.com/c/8089/download-all
 <br>
Make sure to store the training dataset in /data/train directory. Find all information about how the data is structured on this link:
 <br>
https://www.kaggle.com/c/data-science-bowl-2018/data
### Usage
#### Train
Training model based on following parameters
- train_path: The path to the training directory. Default set to ./data/train/
- batch_size: Default set to 16
- epochs: Default set to 20

To run use the following command: 
<br>
mlflow run . -e train -P train_path = ./data/train/ -P batch_size = 16 -P epochs = 20

#### Segment
Segmenting data of your choice.
- input_path: The path to the input file for segmentation. Default set to ./data/sample/sample.png
- model_path: Pretrained model path. Default set to ./TrainedModelsUNET/unet_best.h5

To run use the following command: 
<br>
mlflow run . -e segment -P input_path = ./data/sample/sample.png -P model_path = ./TrainedModelsUNET/unet_best.h5
