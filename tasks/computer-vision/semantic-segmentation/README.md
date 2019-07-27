# Semantic Segmentation
Model for semantic segmentation of nuclei from cell images. The model used is U-Net.
###Usage
####Train
Training model based on following parameters
- train_path: The path to the training directory. Default set to ./data/train/
- batch_size: Default set to 16
- epochs: Default set to 20

To run use the following command: 
<br>
mlflow run . -e train -P train_path = ./data/train/ -P batch_size = 16 -P epochs = 20

####Segment
Segmenting data of your choice.
- input_path: The path to the training directory. Default set to ./data/train/
- model_path: Pretrained model path. Default set to 

To run use the following command: 
<br>
mlflow run . -e train -P train_path = ./data/train/ -P batch_size = 16 -P epochs = 20
