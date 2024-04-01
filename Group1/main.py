import os  # Importing the os module to perform operating system dependent functionality.
import pickle  # Importing pickle module to serialize and deserialize Python objects.

from skimage.io import imread  # Importing imread function from skimage.io to read images.
from skimage.transform import resize  # Importing resize function from skimage.transform to resize images.
import numpy as np  # Importing numpy library for numerical operations.
from sklearn.model_selection import train_test_split  # Importing train_test_split function from sklearn.model_selection to split data into training and testing sets.
from sklearn.model_selection import GridSearchCV  # Importing GridSearchCV class from sklearn.model_selection to perform hyperparameter tuning.
from sklearn.svm import SVC  # Importing SVC class from sklearn.svm to implement Support Vector Classification.
from sklearn.metrics import accuracy_score  # Importing accuracy_score function from sklearn.metrics to evaluate classification accuracy.


# prepare data
input_dir = '/Users/basmala/Documents/GitHub/aiRoadSafety.github.io/Group1/data/train'  # Path to the directory containing training images.
categories = ['Accident', 'Non Accident']  # Categories of images.

data = []  # List to store flattened image data.
labels = []  # List to store corresponding labels.

# Loop through each category and each image file within the category directory.
for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)  # Path to the current image.
        img = imread(img_path)  # Read the image.
        img = resize(img, (15, 15))  # Resize the image to 15x15 pixels.
        data.append(img.flatten())  # Flatten the image and add it to the data list.
        labels.append(category_idx)  # Add the label to the labels list.

data = np.asarray(data)  # Convert data list to numpy array.
labels = np.asarray(labels)  # Convert labels list to numpy array.

# train / test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)  # Split data into training and testing sets.

# train classifier
classifier = SVC()  # Create a Support Vector Classification model.

parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]  # Define a grid of hyperparameters to search for the best combination during grid search.

grid_search = GridSearchCV(classifier, parameters)  # Instantiate GridSearchCV with the SVM classifier and the defined parameters grid.

grid_search.fit(x_train, y_train)  # Fit the grid search model to find the best combination of hyperparameters.

# Test model performances
best_estimator = grid_search.best_estimator_  # Retrieve the best performing model from the grid search results.

y_prediction = best_estimator.predict(x_test)  # Generate predictions on the test set using the best model.

score = accuracy_score(y_prediction, y_test)  # Compute the accuracy score of the predictions.

print('{}% of samples were correctly classified'.format(str(score * 100)))  # Print the accuracy score on the test set.

pickle.dump(best_estimator, open('./model.p', 'wb'))  # Serialize and save the best performing model to a file for future use.
