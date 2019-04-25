# Cancer-Detection

Datasets:
1. HAM10000 - Contains 10000 images of skin cancer with dimension of 100 x 75.
2. SKIN_CANCER_metadata.csv - Contains image names and corresponding labels for the 10000 images.
3. SKIN_CANCER_metadata_mini.csv - Contains filtered data for skin cancer.
4. Skin_Data.zip - Split dataset into training and testing data after preprocessing.
5. Skin_Data_CNN.zip - Split dataset into training and testing data without preprocessing for cnn.
6. breast_cancer_data.csv - Contains the features of the breast tumour cell and the corresponding label.

Files:
1. ImageProcessing.py - Converts the images into pixel values and stores them into mini-batches of given size.
2. merge_data.py - Converts batch files into training and testing data depending on the model. Also, contains functions to fetch the datasets.
3. confusion_matrix.py - Takes target and predicted labels and generates the corresponding confusion matrix.
4. ROC.py - Takes target and predicted labels and generates the corresponding ROC curve.
5. logistic.py - Contains the class that performs Logistic Regression.
6. nn.py - Contains the class that performs classification using Feed Forward Neural Network.
7. cnn.py - Contains the code that performs classification using Convolutional Neural Network on Images dataset.
8. svm.py - Contains the class that performs classification using Support Vector Machine.
9. testing.py - Runs the models provided in the main function and displays accuracy, confusion matrix and ROC curve for each model.
10. Plots - Contains the generated confusion matrix and ROC curves for each model.

Libraries Required:
1. pytorch
2. tensorflow

Preferred python version - 3.6

How to Run:
1. Download the repository in a folder.
2. Extract all the .zip files.
3. Open testing.py and uncomment the model you wish to run in main function.
4. Run testing.py
