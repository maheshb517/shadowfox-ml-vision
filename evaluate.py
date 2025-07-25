import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from dataset import X_test, y_test, class_names

# Load the saved model
model = tf.keras.models.load_model('models/image_classifier.h5')

# Predict on test set
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)

# Show 9 test images with predicted and actual labels
plt.figure(figsize=(10, 5))
for i in range(9):
    index = np.random.randint(0, len(X_test))
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_test[index])
    plt.title(f"Pred: {class_names[predicted_labels[index]]}\nActual: {class_names[true_labels[index]]}")
    plt.axis("off")

plt.tight_layout()
plt.show()
