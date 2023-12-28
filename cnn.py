import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model


model_top_path = "saved_model/CNN"

# Data paths
train_data_path = "Training and Validation"
test_data_path = "Testing"

# Data preprocessing
train_datagen = ImageDataGenerator(rescale=1 / 255)
test_datagen = ImageDataGenerator(rescale=1 / 255)

train_dataset = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(250, 250),
    batch_size=32,
    class_mode='binary'
)

test_dataset = test_datagen.flow_from_directory(
    test_data_path,
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
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(hp.Int('units_1', min_value=32, max_value=256, step=32), (3, 3), activation='relu',
                                input_shape=(250, 250, 3)))
        model.add(layers.MaxPool2D(2, 2))

        for i in range(hp.Int('num_conv_layers', min_value=1, max_value=3)):
            model.add(layers.Conv2D(hp.Int(f'conv_units_{i}', min_value=32, max_value=256, step=32), (3, 3),
                              activation='relu'))
            model.add(layers.MaxPool2D(2, 2))

        model.add(layers.Flatten())
        model.add(layers.Dense(hp.Int('dense_units', min_value=128, max_value=512, step=64), activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # Define the Keras Tuner RandomSearch tuner
    tuner = RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=5,  # Number of hyperparameter combinations to try
        executions_per_trial=2,  # Number of times to train the model per trial
        directory='tuner_results',  # Directory to save results
        project_name='cnn_tuning'  # Name of the tuning project
    )

    # Perform hyperparameter tuning
    tuner.search(train_dataset,
                 epochs=10,  # Number of epochs per trial
                 validation_data=test_dataset)

    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Build the final model with the best hyperparameters
    model = build_model(best_hps)

    log_dir = "logscnn/fit"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    plot_model(model, to_file='model_plotcnn.png', show_shapes=True, show_layer_names=True)
    # Train the model with the best hyperparameters
    history = model.fit(
        train_dataset,
        epochs=100,
        validation_data=test_dataset, callbacks=[tensorboard_callback]
    )


    # Save the model
    model.save(model_top_path)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print("Test accuracy:", test_accuracy)

    # Plot training and validation accuracy
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.legend()
    plt.savefig("saved_model/CNN/accuracy.png")
    plt.show()

# Function to predict and display images
def predict_image(filename):
    img = tf.keras.preprocessing.image.load_img(filename, target_size=(250, 250))
    plt.imshow(img)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    if prediction[0] >= 0.5:
        plt.xlabel("Fire", fontsize=30)
    else:
        plt.xlabel("No Fire", fontsize=30)
    plt.show()

# Predict and display some images
predict_image("../forest_fire/Testing/fire/abc182.jpg")
predict_image("../forest_fire/Testing/nofire/abc361.jpg")
