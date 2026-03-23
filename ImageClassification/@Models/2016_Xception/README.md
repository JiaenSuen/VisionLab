## Reference
### Xception: Deep Learning with Depthwise Separable Convolutions



SeparableConv2d:
Depthwise separable convolution splits spatial and channel mixing.
Depthwise performs spatial filtering per channel, pointwise mixes channels.
This greatly reduces parameters and computation while maintaining performance.

EntryBlock:
Entry flow extracts low-level features and reduces spatial size.
Residual shortcut with stride ensures stable gradient flow.
Separable convs improve efficiency compared to standard convolution.

MiddleBlock:
Middle flow learns complex representations at constant resolution.
Multiple separable conv layers with residual connection improve feature reuse
and enable very deep architecture without degradation.

ExitBlock:
Exit flow expands channel dimension and reduces spatial resolution.
Prepares high-level semantic features for classification.
Residual path ensures stable transition to classifier.

XceptionTiny:
Tiny version adapts Xception for CIFAR resolution.
Reduces channel size, number of middle blocks, and downsampling steps
to avoid excessive information loss on small images.