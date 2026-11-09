# Pet Breed Classification with Explainable AI (Grad-CAM)

A PyTorch image-classification project that predicts the breed of a cat or dog from an image and uses **Grad-CAM** to visualize the image regions that influenced the model's prediction.

The project uses the **Oxford-IIIT Pet Dataset**, a pretrained **ResNet-18** model, and segmentation masks supplied with the dataset to evaluate whether Grad-CAM highlights the pet in the image.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model](#model)
- [Results](#results)
- [Grad-CAM Explainability](#grad-cam-explainability)
- [Project Workflow](#project-workflow)
- [How to Run](#how-to-run)
- [Saving and Loading the Model](#saving-and-loading-the-model)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)

## Project Overview

Image classifiers can predict a label without making it clear which parts of an image influenced that prediction. This project explores both image classification and visual explanation.

Given a cat or dog image, the model:

1. Processes the image at 224 × 224 pixels.
2. Predicts one of 37 pet breeds.
3. Reports the predicted class and its softmax confidence.
4. Generates a Grad-CAM heatmap for the predicted class.
5. Compares high-activation regions with the dataset's pet segmentation mask on a sample of test images.

## Dataset

**Oxford-IIIT Pet Dataset**

The notebook downloads and uses the `trainval` and `test` splits from `torchvision.datasets.OxfordIIITPet`.

| Split | Number of images |
|---|---:|
| Training (`trainval`) | 3,680 |
| Testing (`test`) | 3,669 |
| Number of breed classes | 37 |

The dataset contains 37 cat and dog breeds. The notebook uses the category labels for classification and the segmentation masks for localization evaluation.

### Breed classes

Abyssinian, American Bulldog, American Pit Bull Terrier, Basset Hound, Beagle, Bengal, Birman, Bombay, Boxer, British Shorthair, Chihuahua, Egyptian Mau, English Cocker Spaniel, English Setter, German Shorthaired, Great Pyrenees, Havanese, Japanese Chin, Keeshond, Leonberger, Maine Coon, Miniature Pinscher, Newfoundland, Persian, Pomeranian, Pug, Ragdoll, Russian Blue, Saint Bernard, Samoyed, Scottish Terrier, Shiba Inu, Siamese, Sphynx, Staffordshire Bull Terrier, Wheaten Terrier, and Yorkshire Terrier.

## Model

The notebook contains an initial custom CNN experiment. The later, evaluated classifier uses **ResNet-18 with ImageNet pretrained weights**.

The transfer-learning setup is:

- Load ResNet-18 with its default pretrained weights.
- Freeze the pretrained model parameters.
- Replace the final fully connected layer with a new layer containing 37 outputs.
- Train the replacement layer using the pet-breed training data.
- Use cross-entropy loss and the Adam optimizer with a learning rate of `0.001`.
- Train for 5 epochs.

The images are resized to 224 × 224 pixels. Training images also receive random horizontal flips. Both training and test images are normalized using the ImageNet mean and standard deviation.

## Results

The notebook records the following results for the ResNet-18 experiment:

| Evaluation | Result |
|---|---:|
| Test-set classification accuracy | **86.37%** |
| Test images evaluated for classification report and confusion matrix | 3,669 |
| Grad-CAM localization evaluation | 100 images |
| Classification accuracy on that 100-image localization sample | 79.00% |
| Mean pet localization score on that sample | 79.75% in one recorded run; 78.93% in a later recorded run |

The 100-image localization results are a separate subset evaluation and should not be confused with the full test-set accuracy. The notebook contains two localization runs with slightly different reported mean scores; both are retained here rather than presenting them as one identical result.

The notebook also generates a confusion matrix and a per-class precision, recall, and F1-score report. These are useful for identifying breeds the model distinguishes well and breeds it confuses.

## Grad-CAM Explainability

**Gradient-weighted Class Activation Mapping (Grad-CAM)** creates a heatmap showing image regions that contribute to a selected class score.

In this project:

- Grad-CAM is applied to the final ResNet-18 convolutional block (`model.layer4[-1]`).
- The explanation target is the class predicted by the model.
- The heatmap is overlaid on the original image.
- Example visualizations show the original image, heatmap, and explanation overlay.

### Pet localization evaluation

The Oxford-IIIT Pet Dataset provides segmentation masks. The notebook uses these masks to estimate how much of the strongest Grad-CAM activation falls within the pet region.

The evaluation procedure:

1. Select the top 20% of Grad-CAM activation values.
2. Mark the pet pixels using the segmentation mask (mask labels 1 and 3 in the notebook).
3. Calculate the proportion of selected high-activation pixels that overlap the pet mask.
4. Average the score over 100 test images.

This is a simple localization measure. It indicates whether high-activation regions overlap the pet; it does **not** prove that the model uses only meaningful features or that an explanation is causally faithful.

## Project Workflow

The notebook follows this workflow:

1. Install and import the required libraries.
2. Download and inspect the Oxford-IIIT Pet Dataset.
3. Prepare image transformations and data loaders.
4. Experiment with a custom CNN.
5. Set up pretrained ResNet-18 for transfer learning.
6. Train the classifier and evaluate test accuracy.
7. Generate Grad-CAM visualizations.
8. Test the model on an uploaded image.
9. Save the trained model weights and class names.
10. Generate a confusion matrix and per-class classification report.
11. Evaluate Grad-CAM localization against pet segmentation masks.

## How to Run

### Option 1: Google Colab

1. Upload `Image_recognition_project.ipynb` to Google Colab.
2. Select **Runtime → Change runtime type** and choose a GPU if one is available.
3. Run the notebook cells in order.
4. Allow `torchvision` to download the Oxford-IIIT Pet Dataset when prompted by the dataset code.
5. To test a personal image, run the upload cell and select an image file.

### Option 2: Local Jupyter environment

1. Clone or download this repository.
2. Install the project dependencies using the separate requirements file in this repository.
3. Open the notebook in Jupyter Notebook or JupyterLab.
4. Run the cells in order.

The dataset is downloaded to the notebook's `./data` directory. A GPU is optional.

## Instructions

Install pachages using
pip install -r requirements.txt

## Saving and Loading the Model

The notebook includes code to save the model weights and the breed class names separately. The saved files are:

- `pet_breed_resnet18.pth` — model state dictionary.
- `class_names.json` — ordered list of the 37 breed names.

Keep the class-name order consistent with the model's output indices when loading the model. The notebook also includes a check that the saved files exist and that the class-name list contains 37 classes.

## Limitations

- The classifier predicts one of the 37 breeds in the training dataset. It is not designed to identify breeds outside those classes.
- A high softmax confidence is not a guarantee that a prediction is correct.
- Some visually similar breeds may be confused.
- The reported results come from the notebook's particular training run and may vary with software versions, random seeds, hardware, or training changes.
- The localization score is a basic overlap measure, not a complete evaluation of explanation quality.
- Grad-CAM visualizations are approximate explanations and should not be treated as proof of the model's reasoning.
- The notebook's initial custom CNN experiment performed substantially worse than the later pretrained ResNet-18 experiment; the reported final classifier results refer to ResNet-18.

## Future Improvements

- Fine-tune selected ResNet-18 layers instead of training only the final layer.
- Train for more epochs and compare validation performance.
- Add reproducible random seeds and a validation-based model-selection process.
- Investigate per-breed errors using the confusion matrix and classification report.
- Test on a larger and more varied set of images from outside the dataset.
- Compare Grad-CAM with other explanation methods.
- Improve the localization evaluation and document its assumptions.
- Add example prediction and Grad-CAM images to the repository.

## Acknowledgements

- **Oxford-IIIT Pet Dataset** for the pet images, breed labels, and segmentation masks.
- **PyTorch** and **Torchvision** for the deep-learning framework and pretrained ResNet-18 implementation.
- **pytorch-grad-cam** for Grad-CAM visualization tools.

---

*This repository is an educational project exploring transfer learning, image classification, and visual explainability with Grad-CAM.*
