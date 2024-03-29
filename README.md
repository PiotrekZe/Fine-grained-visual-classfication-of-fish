# Fine grained visual classfication of fish

This repository contains a project prepared for the IVUS2024 conference (29th International Conference on Information Society and University Studies). I developed a convolutional neural network (CNN) model for the problem of fine-grained visual classification. The dataset used is the Croatian Fish Dataset prepared for the Machine Vision of Animals and their Behavior Workshop 2015 (doi: 10.5244/C.29.MVAB.6).

Fine-grained visual classification (FGVC) involves classifying images belonging to the same meta-class. This problem is challenging due to the small differences between classes and the limited amount of data available. In this case, the dataset contains only 794 images of fish belonging to one of twelve classes.

Throughout this project, I had to work with low-resolution data; some images were as small as 30x25 pixels. Additionally, the images were blurry and exhibited a blue shift, a consequence of the nature of the photos taken. To address this, I performed simple image preprocessing: each image was brightened and resized to 64x64 pixels. Furthermore, data augmentation techniques such as horizontal and vertical flipping were applied.

![Image of model](images/before_after_preprocessing.jpg)

I designed a custom CNN model with a CBAM attention module (https://arxiv.org/abs/1807.06521) and skip connections inspired by Densenet (https://pytorch.org/hub/pytorch_vision_densenet/). This model achieved an accuracy of 94.375%. In comparison, the authors of the dataset achieved 66.75% accuracy using a pre-trained ImageNet model for feature extraction and an SVM classifier.

![Image of model](images/Mode.png)


# Test Set Results:
Below are the results of the model on the test set:

![Image of model](images/accuracy_recall_precision_f1.png)
