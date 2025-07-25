# Step 1: Import libraries
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Step 2: Load CIFAR-10 dataset (auto-downloads)
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Step 3: Define class names for the 10 categories
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Step 4: Print dataset shape
print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

# Step 5: Visualize 9 random images with labels
plt.figure(figsize=(10, 5))
for i in range(9):
    index = np.random.randint(0, len(X_train))
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_train[index])
    plt.title(class_names[y_train[index][0]])
    plt.axis("off")
plt.tight_layout()
plt.show()

# Step 6: Normalize the image pixel values to range [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Step 7: Convert labels to one-hot encoded format
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train, 10)  # 10 classes
y_test = to_categorical(y_test, 10)

# Step 8: Confirm new shape of labels
print("y_train shape after one-hot encoding:", y_train.shape)
print("y_test shape after one-hot encoding:", y_test.shape)
