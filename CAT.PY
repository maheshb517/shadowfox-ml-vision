import os
from tensorflow.keras.datasets import cifar10
from PIL import Image

# Load CIFAR-10
(_, _), (X_test, y_test) = cifar10.load_data()

# Save 5 cat images (label 3 = cat)
os.makedirs("exported_cats", exist_ok=True)
cat_count = 0
for i in range(len(X_test)):
    if y_test[i] == 3:
        img = Image.fromarray(X_test[i])
        img.save(f"exported_cats/cat_{cat_count+1}.png")
        cat_count += 1
        if cat_count == 5:
            break
