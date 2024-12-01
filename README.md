# voice-identification-classification-algorithm
Prerequisites
Before running the script, ensure the following libraries are installed:

numpy
pandas
matplotlib
scikit-learn
tensorflow
Install missing libraries using pip:

pip install numpy pandas matplotlib scikit-learn tensorflow

Steps:

1. Data Preparation
Unzipping the Dataset: The dataset is expected to be in a zipped file named voice.csv.zip. Use ZipFile to extract it to a specific directory.
Loading Data: Load the voice.csv dataset using Pandas.
Inspecting the Dataset: Use .info() to check data types, null values, and overall structure.
Encoding Labels: Transform the categorical label column into numeric values using LabelEncoder.
Feature Scaling: Standardize the features using StandardScaler to ensure they are on a similar scale.
Train-Test Split: Split the dataset into training and testing sets (70% training, 30% testing).

2. Fully Connected Neural Network
Defining the Model:
Input layer matches the number of features.
Two hidden layers with 64 neurons each and ReLU activation.
Output layer with a single neuron and sigmoid activation for binary classification.
Compiling the Model:
Optimizer: Adam
Loss Function: binary_crossentropy
Metrics: Accuracy and AUC
Training:
Train the model with EarlyStopping to monitor val_loss and stop training if performance stagnates.
Evaluation:
Evaluate the trained model on the test set.

3. Preparing Data for CNN
Reshaping the Data:
Pad the feature matrix into a uniform shape using pad_sequences.
Reshape data into 3D tensors suitable for a CNN, adding a channel dimension.
Visualizing Data:
Use Matplotlib to display sample data as 2D heatmaps.

4. Convolutional Neural Network (CNN)
Defining the Model:
Input layer matches the reshaped data dimensions.
Two convolutional layers followed by max-pooling:
16 filters, kernel size 2 for the first convolutional layer.
32 filters, kernel size 1 for the second convolutional layer.
Flatten the output from convolution layers.
Dense layer with 64 neurons and ReLU activation.
Output layer with a single neuron and sigmoid activation.
Compiling the Model:
Same optimizer, loss function, and metrics as the fully connected network.
Training:
Train the CNN using a validation split and monitor performance over epochs.
Evaluation:
Evaluate the model's accuracy and AUC on the test set.
