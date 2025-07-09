# This is a personal practice mini-project.

It is inspired by the "Pet Image Classification and Detection with Transformers" project from the "Learn Image Classification with PyTorch" Course offered by Codecademy.

In this project, I classify and locate pets in images using deep learning models based on transformers. The dataset used is the **Oxford-IIIT Pet Dataset**, which contains images of 37 different dog and cat breeds. Each image includes:

1. A species label (`cat` or `dog`)
2. A breed label
3. A bounding box around the pet
4. A segmentation mask (not used in this project)

The project focuses on two main tasks:

### 1. **Image Classification**

I fine-tuned a pre-trained Vision Transformer (ViT) to classify whether an image contains a cat or a dog.

The model was trained using:

* A Vision Transformer (ViT-B/16, pre-trained on ImageNet)
* Binary cross-entropy loss
* Accuracy and class-wise precision/recall/F1 metrics
* Early stopping based on validation performance

My personal best:

```
Classification report:
              precision    recall  f1-score   support

         cat       1.00      1.00      1.00       7
         dog       1.00      1.00      1.00       7

    accuracy                           1.00       14
   macro avg       1.00      1.00      1.00       14
weighted avg       1.00      1.00      1.00       14

```

### 2. **Object Detection**

I used the bounding box annotations to locate the pet in each image. For this task, I evaluated a pre-trained DETR (DEtection TRansformer) model (`facebook/detr-resnet-50`) to see how well it detects and localizes pets.

The detection pipeline involved:

* Loading images and bounding box labels
* Applying DETR for object detection
* Comparing predicted boxes with ground truth using Intersection over Union (IoU)

Example evaluation metrics:

```
📊 Evaluation Results (score_threshold=0.99):
Average IoU: 0.829
Detection Accuracy (IoU > 0.7): 2/2 (100.0%)
Precision: 1.000
Recall:    1.000
F1 Score:  1.000
```

---

This project helped me gain hands-on experience using transformer-based architectures for both classification and object detection in computer vision. It also improved my understanding of data preprocessing, fine-tuning, and evaluation using real-world datasets.

---