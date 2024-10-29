# Music-Classification



## Project Description
This project classifies indigenous Nepali music genres by converting audio files into log spectrograms and training custom **ResNet models**. The ResNets effectively capture features from spectrograms for accurate genre classification. The repository includes all necessary code for data processing and model training.

## Data Access
**Note**: Due to privacy and storage constraints, the dataset used in this project is not included in the repository. However, all code and data preprocessing steps are available for adaptation to other datasets. 

# Table of Contents
1. [Project Description](#project-description)
2. [Requirements](#requirements)
3. [Data Preparation](#data-preparation)
4. [Data Loaders](#data-loaders)
5. [Model Architecture](#model-architecture)
6. [Training and Evaluation](#training-and-evaluation)
7.
8. [How to Use](#how-to-use)
9. [License](#license)
10. [Acknowledgments](#acknowledgments)




## Requirements
To run this project, you need to install the required Python packages. You can do this by running the following command in your terminal:

**pip install -r requirements.txt**


## How to Use
1. **Clone the Repository**: 
   ```bash
   git clone https://github.com/Melina-Singh/Music-Classification.git
   cd Music-Classification





## Data Preparation

1. **Data Collection**: 
   - Videos were initially gathered from free online platforms.
  
2. **Audio Conversion**: 
   - The collected video data was converted into audio format for further processing.

3. **Data Cleaning**: 
   - The audio data was cleaned to ensure quality. 
   - Due to a lack of data for certain genres, data augmentation techniques were employed, including pitch shifting and time stretching.

4. **Audio Chunking**: 
   - Each audio file was cut into 30-second chunks to standardize input lengths for the model.

5. **Manual Selection and Conversion**: 
   - The chunks were manually reviewed and selected for relevance.
   - The selected audio chunks were then converted into log mel spectrograms for input into the classification model.

## Data Loaders 
The data loaders in this project efficiently load and preprocess log mel spectrograms for training and validation. The **SpectrogramDataset** class organizes audio spectrograms into a structured dataset, associating each spectrogram with its respective genre label based on the directory structure. It includes error handling to skip corrupted images and shuffles the files for randomness during training. The **DataLoader** class creates batches of spectrograms, allowing for efficient parallel data loading, while applying necessary transformations such as resizing and normalization to prepare the images for input into the ResNet models.


## Model Architecture

The ResNet Architecture consists of:

BasicBlock: The building block of the ResNet, which includes two convolutional layers with Batch Normalization and ReLU activation. It employs skip connections to enable the flow of gradients, mitigating the vanishing gradient problem.

ResNet: The main class that constructs the network by stacking multiple BasicBlocks. It starts with a convolutional layer, followed by a series of layers with increasing depth, and concludes with a fully connected layer that outputs class probabilities for 35 genres.

Adaptive Pooling: An adaptive average pooling layer is used to ensure the output size is consistent, regardless of the input dimensions.

Dropout Layer: A dropout layer is incorporated to reduce overfitting during training

This architecture effectively captures spectrogram features for accurate genre classification.

![ResNet - 34 Model](https://github.com/user-attachments/assets/9137f21e-d716-4fd9-9f6a-fb84854e220d)


## Training and Evaluation
The model was trained using the Adam optimizer with a learning rate of 0.001 and a CrossEntropyLoss function. The training loop included data augmentation techniques such as pitch shifting and time stretching to enhance model robustness.

### Evaluation Metrics
Model performance was evaluated using accuracy and F1 score on the validation set, providing insights into the classification effectiveness.

### Results
- The model achieved an accuracy of 82.67% on the validation set, demonstrating its capability to classify indigenous Nepali music genres accurately.

### Confusion Matrix
The confusion matrix illustrates the performance of the classification model across different genres, highlighting areas of strength and potential improvement.

![ramroconfusionm](https://github.com/user-attachments/assets/aa1ed195-c414-4eed-99c6-a81b12a6e94a)


### ROC Curve
The ROC curve provides a graphical representation of the model's true positive rate against the false positive rate, indicating the trade-offs between sensitivity and specificity.

![ramro_ROC](https://github.com/user-attachments/assets/9db92795-62d7-4017-b39a-7333729e2576)

### Model Saving and Loading
Instructions for saving and loading the model is provided in the training scripts.


## License
This project is licensed under the GPL License. See the LICENSE file for more information.


## Authors
- **Melina Singh**
- **Rakhsya Bhusal**

##
