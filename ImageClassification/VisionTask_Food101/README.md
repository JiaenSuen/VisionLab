# **Vision Classifier Architectures in Practice : A From-Scratch Experimental Benchmark across Models and Datasets**
## **Overview**
This project is a systematic empirical study of image classification architectures, ranging from early CNN-based models such as VGG and ResNet to modern transformer and SSM-based designs.

Rather than aiming for state-of-the-art performance, the goal is to understand architectural evolution, inductive biases, and training behaviors through from-scratch PyTorch implementations and controlled experiments across multiple datasets.


*Device : NVIDIA GeForce RTX 4060 Laptop GPU(CUDA)* : Inference Time (ms) is for reference.

## **Classifier Models**
### CNN

| 2014      | 2015          |2016           |2017      |2018           |2019          |2021            |
| ---       | ---           | ---           | ---      |---            |---           |---             |
|NIN        |InceptionV2    |InceptionV4    |MobileNet |MobileNetV2    |EfficientNet  |EfficientNetV2  |
|VGG        |InceptionV3    |InceptionResNet|ShuffleNet|ShuffleNetV2   |HRNet         |RegNet          |
|InceptionV1|Highway        |ResNetV2       |DenseNet  |MnasNet        |CSPNet        |MobileNetV3     |
|All-ConvNet|ResNet         |Wide ResNet    |NASNet    |.              |.             |.               |
|.          |.              |Xception       |SENet     |.              |.             |.               |
|.          |.              |ResNeXt        |.         |.              |.             |.               |
|.          |.              |SqueezeNet     |.         |.              |.             |.               |

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


 

### Food101 (with no Data Augmentation)
| Model                     | Train Accuracy | Test Accuracy | Epoches   |  GFLOPs      | Inference Time (ms) |  
| ------------------------- | -------------  | ------------- | --------- |------------- | ------------------- | 
| ResNet18                  | 93.65          | 65.65         | 30        | 3.6271       | 0.6916
| E-ConvNeXt-mini           | --.--          | 67.65         | 10        | 1.2301       | 0.5642


## Addendum : Models Research
### Part 1
* NIN : Network In Network [Dec 2013]
* VGG : Very Deep Convolutional Networks for Large-Scale Image Recognition [Sep 2014]
* GoogLeNet (Inception-V1) : Going Deeper with Convolutions [Sep 2014]
* All-ConvNet : Striving for Simplicity: The All Convolutional Net [Dec 2014]
* Batch Normalization : Accelerating Deep Network Training by Reducing Internal Covariate Shift [Feb 2015]
* InceptionV2 : Rethinking the Inception Architecture for Computer Vision [Dec 2015]
* InceptionV3 : Rethinking the Inception Architecture for Computer Vision [Dec 2015]
* Highway Network : Highway Networks [May 2015]
* ResNet : Deep Residual Learning for Image Recognition [Dec 2015]
* InceptionV4 : Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning [Feb 2016]
* ResNet-v2 : Identity Mappings in Deep Residual Networks [Mar 2016]
* Wide ResNet : Wide Residual Networks [May 2016]
* Xception : Xception: Deep Learning with Depthwise Separable Convolutions [Oct 2016]
* ResNeXt : Aggregated Residual Transformations for Deep Neural Networks [Nov 2016]
### Part 2
* SqueezeNet : AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size [Feb 2016]
* MobileNets : Efficient Convolutional Neural Networks for Mobile Vision Applications [Apr 2017]
* ShuffleNet : An Extremely Efficient Convolutional Neural Network for Mobile Devices [Jul 2017]
* DenseNet : Densely Connected Convolutional Networks [Aug 2016]
* NASNet :Learning Transferable Architectures for Scalable Image Recognition [Jul 2017]
* SENet : Squeeze-and-Excitation Networks [Sep 2017]
* MobileNetV2 : Inverted Residuals and Linear Bottlenecks [Jan 2018]
* ShuffleNetV2 : Practical Guidelines for Efficient CNN Architecture Design [Jul 2018]
* MnasNet : Platform-Aware Neural Architecture Search for Mobile [Jul 2018]
* EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks [May 2019]
* HRNet : Deep High-Resolution Representation Learning for Visual Recognition [Aug 2019]
* CSPNet : A New Backbone that can Enhance Learning Capability of CNN [Nov 2019]
* EfficientNetV2 : Smaller Models and Faster Training [Apr 2021]
* RegNet : Self-Regulated Network for Image Classification [Jan 2021]
* MobileNetV3 : Searching for MobileNetV3 [May 2019]