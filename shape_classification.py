import os
import numpy as np
import cv2
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.utils import shuffle




class NNN:
    def params(self):
        # Xavier Inilization of weights for better 
        # Changed np.random.randn() to .rand() and the accuracy skyrocketed. The training set got 96%. The testing set got 88%
        self.w1 = np.random.rand(10, 784) * np.sqrt(1/784)
        self.b1 = np.random.rand(10, 1) - 0.5
        self.w2 = np.random.rand(3, 10) * np.sqrt(2/10)
        self.b2 = np.random.rand(3, 1) - 0.5
        

    def get_predictions(self,a2):
        return np.argmax(a2, axis=0)

    def accuracy(self, y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / y_true.size
        return accuracy

    def relu(self, z):
        return np.maximum(0, z)

    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True)

    def derivative_relu(self,z):
        return np.where(z > 0, 1.0, 0.0)

    def one_hot(self, y):
        one_hot_y = np.zeros((y.size, 3))
        one_hot_y[np.arange(y.size), y] = 1
        return one_hot_y.T



    def forward(self, x):
        z1 = self.w1 @ x + self.b1
        a1 = self.relu(z1)
        z2 = self.w2 @ a1 + self.b2
        a2 = self.softmax(z2) # (3, 16004)
        return z1, a1, z2, a2

    def backward(self, x, y, z1, a1, a2):
        one_hot_y = self.one_hot(y)
        m = y.size
        Dz2 = a2 - one_hot_y
        Dw2 = 1/m * np.dot(Dz2, a1.T)
        Db2 = 1/m * np.sum(Dz2, axis=1, keepdims=True)
        Dz1 = np.dot(self.w2.T, Dz2) * self.derivative_relu(z1)
        Dw1 = 1/m * np.dot(Dz1, x.T)
        Db1 = 1/m * np.sum(Dz1, axis=1, keepdims=True)
        return Dw1, Db1, Dw2, Db2

    def update(self, Dw1, Db1, Dw2, Db2, alpha):
        self.w1 = self.w1 - alpha * Dw1
        self.b1 = self.b1 - alpha * Db1
        self.w2 = self.w2 - alpha * Dw2
        self.b2 = self.b2 - alpha * Db2


    def train(self, x, y, alpha, epochs, batch_size):
        self.params() # initialize weights and biases
        m = x.shape[1]

        for epoch in range(epochs):


            indices = np.random.permutation(m)
            x_s = x[:, indices]
            y_s = y[:, indices]


            for start in range(0, m, batch_size):
                end = start + batch_size
                x_b = x_s[:,start:end]
                y_b = y_s[:,start:end]
    
                z1, a1, z2, a2 = self.forward(x_b)
                Dw1, Db1, Dw2, Db2 = self.backward(x_b, y_b, z1, a1, a2)
                self.update(Dw1, Db1, Dw2, Db2,alpha)
            if epoch % 100 == 0:
                z1, a1, z2, a2 = self.forward(x)
                pred = self.get_predictions(a2)
                acc = self.accuracy(y, pred)
                print(pred)
                print(f"Epoch {epoch}, Acc {acc}")

    def test(self, x, y):
        z1, a1, z2, a2 = self.forward(x)
        pred = self.get_predictions(a2)
        print("Pred", pred, "True",y)
        print(f"Accuracy: {self.accuracy(y, pred)}")
        return pred




    
# Preprocessing    
def load_shapes():
    """
    Load the paths of all shape images in the dataset.

    Returns a dictionary with the keys 'ellipse', 'rectangle', and 'triangle'.
    Each value is a list of paths to the images of the corresponding shape.
    """
    path = "C:/Users/NR_Se/OneDrive/Documents/GitHub/Final_Project_Machine_Learning/hand-drawn-shapes-dataset-main/data/"
    shapes = ["ellipse", "rectangle", "triangle"]
    data = { "ellipse": [], "rectangle": [], "triangle": []}
    for user in users:
        for shape in shapes:
            file_path = Path(path + user) / "images" / shape
            if file_path.exists():
                data[shape].extend(str(file) for file in file_path.glob("*.png"))
                
    return data

def preprocess_image(file_paths):
    """
    Preprocesses a list of image file paths by reading, resizing, and normalizing them.

    Args:
        file_paths (list of str): List of file paths to the images to be processed.

    Returns:
        np.ndarray: A numpy array of processed images, where each image is resized 
        to 28x28 pixels and pixel values are normalized to the range [0, 1].
    """
    processed_images = []

    for file in file_paths:
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            image = cv2.resize(image, (28, 28))
            image = image / 255
            processed_images.append(image)
    # print(len(processed_images))
    return np.array(processed_images)

def get_labels(file_paths):
    """
    Gets the labels corresponding to the given file paths.

    Args:
        file_paths (dict): A dictionary with the keys 'ellipse', 'rectangle', and 'triangle'.
            Each value is a list of file paths to the images of the corresponding shape.

    Returns:
        np.ndarray: A numpy array of labels, where 0 corresponds to ellipse, 1 to rectangle, and 2 to triangle.
    """
    labels = []
    shape_to_label = { "ellipse": 0, "rectangle": 1, "triangle": 2}
    for shape, files in file_paths.items():
        labels.extend([shape_to_label[shape]] * len(files))

    # print(len(labels))
    return np.array(labels)



users = []
with open("users.txt", "r") as file:
    for line in file:
        users.extend(line.split())

shape_data = load_shapes()

ellipse_data = preprocess_image(shape_data['ellipse'])
rectangle_data = preprocess_image(shape_data['rectangle'])
triangle_data = preprocess_image(shape_data['triangle'])

x_data = np.concatenate((ellipse_data, rectangle_data, triangle_data), axis=0) # Combine all shape data
y_data = get_labels(shape_data)

# Shuffle the data 
x_data, y_data = shuffle(x_data, y_data) 

# Reshape the data
y_data = y_data.reshape(-1, 1) # (20005, 1)
x_data = x_data.reshape(-1, 784) #(20005, 784)


# # Plot the class distribution
# labels = np.unique(y_data)
# values = np.bincount(y_data.ravel().astype(int))

# plt.pie(values, labels=labels, autopct='%1.1f%%')
# plt.title('Class Distribution')
# plt.legend(title='Classes', bbox_to_anchor=(1.2, 0.5), loc='center right')
# plt.show()


# Split the data
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)


x_train = x_train.T # (784, 16004)
x_test = x_test.T # (784, 4001)
y_train = y_train.T # (1, 16004)
y_test = y_test.T # (1, 4001)





#Uncomment to use the NN
# model = NNN()

# # Train the model 
# model.train(x_train, y_train, 0.005, 2000, 32) # best stats alpha = .005, epochs = 2000, batch_size = 32 Accuracy testing set: 88%

#Time to Train: 3 mins and 51 seconds to train

# # Test the model
# y_pred = model.test(x_test, y_test)
# y_test = y_test.reshape(-1) # (4001, )

# # Classification report to evaluate precision, recall, F1 score, etc.
# print(classification_report(y_test, y_pred))

# # Confusion matrix to visualize misclassifications
# cm = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ellipse", "Rectangle", "Triangle"])
# disp.plot(cmap=plt.cm.Blues)
# plt.title("Neural Network Confusion Matrix")
# plt.show()


# Uncomment to use the random forest model
# # Preps data for Random Forest
# x_train_rf = x_train.T # (16004, 784)
# x_test_rf = x_test.T

# y_train_rf = y_train.T
# y_test_rf = y_test.T


# # Create the Random Forest classifier
# rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# # Train the classifier
# rf_classifier.fit(x_train_rf, y_train_rf)

# y_pred_rf = rf_classifier.predict(x_test_rf)

# y_pred1_rf = rf_classifier.predict(x_train_rf)

# print("Training Score", accuracy_score(y_train_rf, y_pred1_rf), "Testing Score", accuracy_score(y_test_rf, y_pred_rf))

# # Classification report to evaluate precision, recall, F1 score, etc.
# print(classification_report(y_test_rf, y_pred_rf))

# # Confusion matrix to visualize misclassifications
# cm = confusion_matrix(y_test_rf, y_pred_rf)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ellipse", "Rectangle", "Triangle"])
# disp.plot(cmap=plt.cm.Blues)
# plt.title("Random Forest Confusion Matrix")
# plt.show()










   

    


        
    

    

