import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from kerastuner.tuners import RandomSearch
import matplotlib.pyplot as plt
import ssl
import os

ssl._create_default_https_context = ssl._create_unverified_context

# Define paths for saving and loading the model
model_top_path = "saved_model/vgg16_model_top"

# Data preprocessing
train_datagen = ImageDataGenerator(rescale=1 / 255)
test_datagen = ImageDataGenerator(rescale=1 / 255)

train_dataset = train_datagen.flow_from_directory(
    "Training and Validation",
    target_size=(250, 250),
    batch_size=32,
    class_mode='binary'
)

test_dataset = test_datagen.flow_from_directory(
    "Testing",
    target_size=(250, 250),
    batch_size=32,
    class_mode='binary'
)


# Check if the model is present in the specified directory
if os.path.exists(model_top_path):
    print("Loading existing model...")
    model = tf.keras.models.load_model(model_top_path)
else:
    print("Model not found, building a new one...")

    # Function to build the model for hyperparameter tuning
    def build_model(hp):
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(250, 250, 3))
        base_model.trainable = False

        model = keras.Sequential([
            base_model,
            keras.layers.Flatten()
        ])

        # Tune the number of units in the dense layer
        hp_units = hp.Int('units', min_value=128, max_value=512, step=64)
        model.add(keras.layers.Dense(units=hp_units, activation='relu'))

        model.add(keras.layers.Dense(1, activation='sigmoid'))

        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=hp_learning_rate),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        return model


    # Define the Keras Tuner RandomSearch tuner
    tuner = RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=5,  # Number of hyperparameter combinations to try
        executions_per_trial=2,  # Number of times to train the model per trial
        directory='tuner_results',  # Directory to save results
        project_name='vgg16_tuning'  # Name of the tuning project
    )


    # Perform hyperparameter tuning
    tuner.search(train_dataset,
                 epochs=10,  # Number of epochs per trial
                 validation_data=test_dataset)

    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Build the final model with the best hyperparameters
    model = build_model(best_hps)
    # Generate a TensorBoard log
    log_dir = "logsVGG/fit"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    plot_model(model, to_file='model_plotvgg.png', show_shapes=True, show_layer_names=True)
    # Train the model with the best hyperparameters
    history = model.fit(
        train_dataset,
        epochs=100,
        validation_data=test_dataset, callbacks=[tensorboard_callback]
    )

    # Save the model
    tf.keras.models.save_model(model, filepath=model_top_path, save_format='tf')


    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print("Test accuracy:", test_accuracy)

    # Plot training and validation accuracy
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.legend()
    plt.savefig("saved_model/vgg16_model_top/accuracy.png")
    plt.show()

# Function to predict and display images
def predict_image(filename):
    img = keras.preprocessing.image.load_img(filename, target_size=(250, 250))
    plt.imshow(img)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    if prediction[0] >= 0.5:
        plt.xlabel("Fire", fontsize=30)
    else:
        plt.xlabel("No Fire", fontsize=30)

# Predict and display some images
predict_image("../forest_fire/Testing/fire/abc182.jpg")
plt.show()
predict_image("../forest_fire/Testing/nofire/abc361.jpg")
plt.show()
