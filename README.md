
---

# Hand Digit Recognition using Keras and OpenCV

## Overview

This project demonstrates hand digit recognition using machine learning techniques, particularly deep neural networks implemented with Keras, and image processing with OpenCV. The goal is to classify hand-drawn digits (from the MNIST dataset) into the correct numeric values.

## Requirements

- Python 3.x
- OpenCV
- Keras
- TensorFlow
- scikit-learn

## Getting Started

1. **Clone this repository:**

   ```shell
   git clone https://github.com/yourusername/hand_digit_recognition.git
   ```

2. **Navigate to the project directory:**

   ```shell
   cd hand_digit_recognition
   ```

3. **Install the required packages:**

   ```shell
   pip install requirement.txt
   ```

4. **Run the `main.py` script:**

   ```shell
   python main.py
   ```

## Code Structure

- **`main.py`**: The main script that orchestrates training and testing of hand digit recognition models.
- **`HandDigit.py`**: A Python class that encapsulates the functionality for loading data, building models, training models, and testing images.
- **`data/mnist_png/`**: The directory containing the MNIST dataset in PNG format.

## How It Works

### Data Loading

Images and labels are loaded from the MNIST dataset stored as PNG images using OpenCV. The images are converted into NumPy arrays.

```python
# Load images and labels
def png_npy(file_path):
    images = []
    labels = []
    for root, dirs, files in os.walk(file_path):
        for filename in files:
            if filename.endswith('.png'):
                image = cv2.imread(os.path.join(root, filename), cv2.IMREAD_GRAYSCALE)
                label = int(os.path.basename(root))
                images.append(image)
                labels.append(label)
    return np.array(images, dtype=float), np.array(labels, dtype=float)
```

### Data Preprocessing

Images are reshaped into a vector form suitable for model input. Pixel values are normalized to a range between 0 and 1.

```python
# Preprocess images
def matrix_image_vector(self):
    self.train_images_vector = self.train_images.reshape(self.train_images.shape[0], -1)
    self.test_images_vector = self.test_images.reshape(self.test_images.shape[0], -1)

def normalize(self):
    self.train_images_vector = np.array(self.train_images_vector, dtype=float)
    self.test_images_vector = np.array(self.test_images_vector, dtype=float)
    for i in range(len(self.train_images_vector)):
        self.train_images_vector[i] = self.train_images_vector[i] / 255.0
    for i in range(len(self.test_images_vector)):
        self.test_images_vector[i] = self.test_images_vector[i] / 255.0
```

### Neural Network Models

Several neural network models with varying architectures are defined using Keras.

```python
# Build neural network models
def build_models():
    tf.random.set_seed(20)

    model_1 = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(25, activation='relu'),
            tf.keras.layers.Dense(15, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ],
        name='model_1'
    )

    model_2 = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(20, activation='relu'),
            tf.keras.layers.Dense(12, activation='relu'),
            tf.keras.layers.Dense(12, activation='relu'),
            tf.keras.layers.Dense(20, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ],
        name='model_2'
    )

    # Define more models here...

    model_list = [model_1, model_2, ...]  # Add more models as needed
    return model_list
```

### Model Training

Models are compiled with loss, optimizer, and metrics configurations. Training is performed on the training dataset for a specified number of epochs.

```python
# Train neural network models
def train_models(self):
    nn_models = self.build_models()
    counter = 1
    for model in nn_models:
        # Compile the model
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            metrics=['accuracy']
        )

        # Train the model
        model.fit(
            self.train_images_vector, self.train_labels,
            epochs=200,
        )

        # Save the trained model
        model.save("model" + str(counter) + ".h5")
        counter += 1
```

### Model Evaluation

Models are evaluated on both the training and cross-validation datasets to calculate classification errors.

```python
# Calculate model errors
def calculate_model_errors(self):
    for model in self.trained_models:
        loss, accuracy = model.evaluate(x=self.train_images_vector, y=self.train_labels)
        train_error = [loss, accuracy]
        self.nn_train_error.append(train_error)

        loss, accuracy = model.evaluate(x=self.cross_images_vector, y=self.cross_labels)
        cv_error = [loss, accuracy]
        self.nn_cv_error.append(cv_error)
        self.errors.append([cv_error, train_error])
```

### Selecting the Best Model

The model with the lowest cross-validation error is selected as the best model.

```python
#

 Select the best model
def select_best_model(self):
    error_model_pairs = list(zip(self.nn_cv_error, self.trained_models))

    # Sort models by cross-validation error
    error_model_pairs.sort(
        key=lambda x: (x[0][0], x[0][1]))  # Sort by loss, then accuracy

    # Get the model with the lowest cross-validation error
    self.best_model = error_model_pairs[0][1]

    # Find the index of the best model in self.trained_models
    self.best_model_index = self.trained_models.index(self.best_model)
```

### Testing

The best model is used to classify hand-drawn digit images.

```python
# Test an image using the best model
def test(self, group, member):
    script_dir = os.path.dirname(__file__)
    mnist_test_dir = os.path.join(script_dir, 'data', 'mnist_png', 'testing')
    image_path = mnist_test_dir + f"/{group}/{member}.png"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    vector_image = image.reshape(1, -1)
    res = self.best_model.predict(vector_image)
    index_of_1 = res.argmax()
    print(index_of_1)
```

## Results

- Classification errors for each model on training and cross-validation sets are displayed.
- The index of the best model and classification errors on the training, cross-validation, and test sets are printed.
- An example test image is classified using the best model, and the result is shown.

