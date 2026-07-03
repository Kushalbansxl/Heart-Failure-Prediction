import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import layers, models
from utils import create_tf_dataset, load_data
from constants import *

def main():
    print("Loading and preprocessing image data...")
    data, labels = load_data(IMG_DIR, CATEGORIES)
    X_train_img, X_temp, y_train_img, y_temp = train_test_split(
        data, labels, test_size=0.3, random_state=42
    )
    X_val_img, X_test_img, y_val_img, y_test_img = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    print("Image data loaded and split successfully.")

    train_img = create_tf_dataset(X_train_img, y_train_img)
    val_img = create_tf_dataset(X_val_img, y_val_img)
    test_img = create_tf_dataset(X_test_img, y_test_img)
    print("TensorFlow datasets created for training, validation, and testing.")

    print("Starting CNN model training...")
    cnn_model = models.Sequential(
        [
            layers.InputLayer(shape=(224, 224, 1)),
            layers.Conv2D(8, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D( 16, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(32, activation="relu"),
            layers.Dense(len(CATEGORIES), activation="softmax"),
        ]
    )

    cnn_model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        str(CNN_MODEL), monitor="val_accuracy", save_best_only=True, mode="max"
    )

    cnn_model.fit(train_img, validation_data=val_img, epochs=10, callbacks=[checkpoint])
    print("CNN model training completed and best model saved.")

    print("Evaluating CNN model...")
    best_model = tf.keras.models.load_model(str(CNN_MODEL))
    cnn_eval = best_model.evaluate(test_img)
    cnn_scores = {
        "loss": cnn_eval[0],
        "accuracy": cnn_eval[1],
    }
    print(f"CNN Evaluation Scores: {cnn_scores}")

    print("Saving CNN evaluation scores...")
    with open(CNN_SCORE, "w") as cnn_score_file:
        json.dump(cnn_scores, cnn_score_file)
    print("CNN scores saved successfully.")

if __name__ == "__main__":
    main()
