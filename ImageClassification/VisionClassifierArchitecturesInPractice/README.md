# **Vision Classifier Architectures in Practice : A From-Scratch Experimental Benchmark across Models and Datasets**
## **Overview**
This project is a systematic empirical study of image classification architectures, ranging from early CNN-based models such as VGG and ResNet to modern transformer and SSM-based designs.

Rather than aiming for state-of-the-art performance, the goal is to understand architectural evolution, inductive biases, and training behaviors through from-scratch PyTorch implementations and controlled experiments across multiple datasets.


## **Classifier Models**
### CNN

| 2014      | 2015          |2016        |2017      |2018           |2019          |2021    |
| ---       | ---           | ---        | ---      |---            |---           |---     |
|NIN        |InceptionV2    |InceptionV4 |DenseNet  |MobileNetV2    |EfficientNet  |RegNet  |
|VGG        |InceptionV3    |ResNetV2    |ShuffleNet|ShuffleNetV2   |HRNet         |.       |
|InceptionV1|Highway        |Wide ResNet |MobileNet |MnasNet        |CSPNet        |.       |
|All-ConvNet|ResNet         |Xception    |NASNet    |.              |.             |.       |
|.          |.              |ResNeXt     |SENet     |.              |.             |.       |
|.          |.              |SqueezeNet  |.         |.              |.             |.       |

### Vision-Transformer /  CNN-Transformer
| 2020 | 2021 |2022 |2023 |2024 |2025|
| ---| ---  | --- | --- |--- |--- |
|ViT|Swin|ConvNeXt|ConvNeXt V2|MobileViT|Hybrid ViT-CNN|
|MLP-Mixer|EfficientNetV2|DaViT|DINOv2|EfficientViT|DINOv3
 

 

### Mamba-based
* Vision Mamba / SSM
* MambaVision
* Hybrid Conv + SSM
* Res-VMamba
* C-RADIOv4


## Datasets 
* STL-10 
* Caltech-101/Caltech-256 
* CIFAR-10 / CIFAR-100 
* Oxford 102 Flowers 
* Stanford Dogs  
* CUB-200 
* Imagenette 
* Imagewoof 
* MedMNIST
 

### CIFAR-10 (with no Data Augmentation)
CIFAR-10: 60,000 32x32 color images in 10 classes (50,000 train, 10,000 test) .
Record Accuracy , Epochs , *GFLOPs Per Image*  <br><br>
(P)   :   PyTorch Pretrained Model API<br>
The following models are included : <br>  

| Model          | Train Accuracy | Test Accuracy | Epoches   |  GFLOPs      |  
| -------------  | -------------  | ------------- | ----------|------------- | 
| GoogLeNet      | 95.16%         | 68.29%        | 200       |0.0496
| GoogLeNet (P)  | 99.63%         | 82.87%        | 100       |0.0611
| InceptionV2    | 99.01%         | 72.91%        | 100       |0.0809
| InceptionV3    | XX.XX%         | XX.XX%        | 100       |
| HighwayNet 23  | 100.00%        | 76.96%        | 100       |0.0969
| ResNet18       |                |               | 100       |
| ResNet18 (P)   | 99.49%         | 86.01%        | 100       |0.2817
| ResNet34 (P)   | 99.30%         | 84.54%        | 100       |0.5837
| Xception       | XX.XX%         | XX.XX%        | 100       |
| SqueezeNet     | XX.XX%         | XX.XX%        | 100       |
| DenseNet121    | XX.XX%         | XX.XX%        | 0         |






## Addendum : Models Research
* NIN : Network In Network [Dec 2013]
* VGG : Very Deep Convolutional Networks for Large-Scale Image Recognition [Sep 2014]
* GoogLeNet (Inception-V1) : Going Deeper with Convolutions [Sep 2014]
* All-ConvNet : Striving for Simplicity: The All Convolutional Net [Dec 2014]
* InceptionV2 : Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift [Feb 2015]
* InceptionV3 : Rethinking the Inception Architecture for Computer Vision [Dec 2015]
* Highway Network : Highway Networks [May 2015]
* ResNet : Deep Residual Learning for Image Recognition [Dec 2015]
* InceptionV4 : Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning [Feb 2016]
* ResNet-v2 : Identity Mappings in Deep Residual Networks [Mar 2016]
* Wide ResNet : Wide Residual Networks [May 2016]
* Xception : Xception: Deep Learning with Depthwise Separable Convolutions [Oct 2016]
* ResNeXt : Aggregated Residual Transformations for Deep Neural Networks [Nov 2016]
* SqueezeNet : AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size [Feb 2016]