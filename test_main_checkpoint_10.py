import tensorflow as tf
from constants import *
from utils import load_data, create_tf_dataset
from sklearn.model_selection import train_test_split
from keras import layers, models

data, labels = load_data(IMG_DIR, CATEGORIES)
X_train_img, X_temp, y_train_img, y_temp = train_test_split(
    data, labels, test_size=0.3, random_state=42
)
X_val_img, X_test_img, y_val_img, y_test_img = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)
train_img = create_tf_dataset(X_train_img, y_train_img)
val_img = create_tf_dataset(X_val_img, y_val_img)

cnn_model = models.Sequential(
    [
        layers.InputLayer(input_shape=(224, 224, 1)),
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
print("Starting training with ModelCheckpoint and full architecture...")
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    str(CNN_MODEL), monitor="val_accuracy", save_best_only=True, mode="max"
)
cnn_model.fit(train_img, validation_data=val_img, epochs=10, callbacks=[checkpoint])
