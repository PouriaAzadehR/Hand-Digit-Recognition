import cv2
import keras
import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split


class HandDigit:

    def __init__(self, first_time):
        mnist_train_dir, mnist_test_dir, mnist_npz = self.give_files_name()
        if first_time:
            self.train_images, self.train_labels = self.png_npy(mnist_train_dir)
            self.test_images, self.test_labels = self.png_npy(mnist_test_dir)
            self.save_multiple_npy()
        else:
            loaded_data = np.load(mnist_npz)
            self.train_images, self.train_labels = loaded_data['train'], loaded_data['train_labels']
            self.test_images, self.test_labels = loaded_data['test'], loaded_data['test_labels']
        self.train_images_vector = []
        self.test_images_vector = []
        self.matrix_image_vector()
        self.normalize()

        self.cross_images_vector = []
        self.cross_labels = []
        self.test_images_vector, self.cross_images_vector, self.test_labels, self.cross_labels = (
            train_test_split(self.test_images_vector, self.test_labels, test_size=0.50,
                             random_state=5))  # 0.5: 50% remained, 50% used
        self.trained_models = []
        self.threshold = 0.5
        self.nn_train_error = []
        self.nn_cv_error = []
        self.errors = []
        self.best_model_index = 0
        self.best_model = None

    # Function to load MNIST data using OpenCV
    @staticmethod
    def png_npy(file_path):
        images = []
        labels = []
        for root, dirs, files in os.walk(file_path):

            for filename in files:
                if filename.endswith('.png'):  # Assuming MNIST images are in PNG format
                    image = cv2.imread(os.path.join(root, filename), cv2.IMREAD_GRAYSCALE)
                    label = int(os.path.basename(root))
                    images.append(image)
                    labels.append(label)
        return np.array(images, dtype=float), np.array(labels, dtype=float)

    # just a test function
    def npy_png(self):
        mnist_npz = os.path.dirname(__file__)
        print(type(mnist_npz))
        for image in self.train_images:
            cv2.imwrite(mnist_npz + 'output_image.png', image)
            break

    @staticmethod
    def give_files_name():
        script_dir = os.path.dirname(__file__)
        mnist_train_dir = os.path.join(script_dir, 'data', 'mnist_png', 'training')
        mnist_test_dir = os.path.join(script_dir, 'data', 'mnist_png', 'testing')
        mnist_npz = os.path.dirname(__file__) + '/mnist.npz'
        return mnist_train_dir, mnist_test_dir, mnist_npz

    def save_multiple_npy(self):
        output_npz_file = os.path.dirname(__file__) + '/mnist.npz'
        print(output_npz_file)
        # Save the arrays into a single .npz file
        np.savez(output_npz_file, train=self.train_images, train_labels=self.train_labels, test=self.test_images,
                 test_labels=self.test_labels)

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

    @staticmethod
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

        model_3 = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(8, activation='relu'),
                tf.keras.layers.Dense(4, activation='relu'),
                tf.keras.layers.Dense(12, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax')
            ],
            name='model_3'
        )
        model_4 = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(5, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax')
            ],
            name='model_4'
        )

        model_list = [model_1, model_2, model_3, model_4]
        return model_list

    def train_models(self):
        nn_models = self.build_models()
        counter = 1
        for model in nn_models:
            # Setup the loss and optimizer
            model.compile(
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                metrics=['accuracy']
            )
            print(f"Training {model.name}...")
            # Train the model
            model.fit(
                self.train_images_vector, self.train_labels,
                epochs=200,
            )
            self.trained_models.append(model)
            model.save("model" + str(counter) + ".h5")
            counter += 1
            print("Done!\n")

    def calculate_model_errors(self):
        for model in self.trained_models:
            loss, accuracy = model.evaluate(x=self.train_images_vector, y=self.train_labels)
            train_error = [loss, accuracy]
            self.nn_train_error.append(train_error)

            loss, accuracy = model.evaluate(x=self.cross_images_vector, y=self.cross_labels)
            cv_error = [loss, accuracy]
            self.nn_cv_error.append(cv_error)
            self.errors.append([cv_error, train_error])

    def select_best_model(self):
        error_model_pairs = list(zip(self.nn_cv_error, self.trained_models))
        print(error_model_pairs)
        # Step 2: Sort models by cross-validation error
        error_model_pairs.sort(
            key=lambda x: (x[0][0], x[0][1]))  # Sort by loss, then accuracy

        # Step 3: Get the model with the lowest cross-validation error
        self.best_model = error_model_pairs[0][1]

        # Step 4: Find the index of the best model in self.trained_models
        self.best_model_index = self.trained_models.index(self.best_model)

    def print_all_model_results(self):
        for model_num in range(len(self.nn_train_error)):
            print(
                f"Model {model_num + 1}: Training Set Classification Error: {self.nn_train_error[model_num][0]}, " +
                f"CV Set Classification Error: {self.nn_cv_error[model_num][0]}"
            )

    def print_train_cross_test_best_model(self):
        loss, accuracy = self.best_model.evaluate(x=self.test_images_vector, y=self.test_labels)
        test_error = [loss, accuracy]

        print("\n" + f"Selected Model Index: {self.best_model_index}")
        print(f"Training Set Classification Error: {self.nn_train_error[self.best_model_index][0]}")
        print(f"CV Set Classification Error: {self.nn_cv_error[self.best_model_index][0]}")
        print(f"Test Set Classification Error: {test_error} \n")

    def load_models(self):
        script_dir = os.path.dirname(__file__)
        for root, dirs, files in os.walk(script_dir):
            for filename in files:
                if filename.endswith('.h5'):  # Assuming MNIST images are in PNG format
                    model = tf.keras.models.load_model(os.path.join(root, filename))
                    self.trained_models.append(model)

    def test(self, group, member):
        script_dir = os.path.dirname(__file__)
        mnist_test_dir = os.path.join(script_dir, 'data', 'mnist_png', 'testing')
        image_path = mnist_test_dir + f"/{group}/{member}.png"
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        vector_image = image.reshape(1, -1)
        res = self.best_model.predict(vector_image)
        index_of_1 = res.argmax()
        print(index_of_1)


handDigit = HandDigit(first_time=False)
handDigit.load_models()
handDigit.calculate_model_errors()
handDigit.select_best_model()
handDigit.print_all_model_results()
handDigit.print_train_cross_test_best_model()

handDigit.test(0, 0)
