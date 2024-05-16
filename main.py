import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Function to create data generators
def create_data_generators(data_path, target_size=(224, 224), batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        data_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        data_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, validation_generator


# Function to build the model
def build_model(input_shape, num_classes):
    # MobileNetV2
    base_model_mobilenet = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model_mobilenet.trainable = False

    # EfficientNetB0
    base_model_efficientnet = EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model_efficientnet.trainable = False

    # ResNet50
    base_model_resnet50 = ResNet50(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model_resnet50.trainable = False

    # Create a model with multiple inputs
    input_tensor = tf.keras.layers.Input(shape=input_shape)
    mobilenet_output = base_model_mobilenet(input_tensor)
    efficientnet_output = base_model_efficientnet(input_tensor)
    resnet50_output = base_model_resnet50(input_tensor)

    # Concatenate the outputs
    merged_output = layers.Concatenate()([mobilenet_output, efficientnet_output, resnet50_output])
    global_average_pooling = layers.GlobalAveragePooling2D()(merged_output)
    dense_layer = layers.Dense(128, activation='relu')(global_average_pooling)
    dropout_layer = layers.Dropout(0.5)(dense_layer)
    output_layer = layers.Dense(num_classes, activation='softmax')(dropout_layer)

    # Create the model
    model = models.Model(inputs=input_tensor, outputs=output_layer)

    return model


# Main training and evaluation loop
def train_and_evaluate(dataset_path, input_shape=(224, 224, 3), batch_size=32, epochs=50):
    # Create data generators
    train_generator, validation_generator = create_data_generators(dataset_path, target_size=input_shape[:2],
                                                                   batch_size=batch_size)
    num_classes = len(train_generator.class_indices)

    # Build the model
    model = build_model(input_shape, num_classes)

    # Compile the model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the model with early stopping
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]
    )

    # Plot accuracy and loss
    plt.figure(figsize=(12, 4))
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Evaluate the model
    evaluation = model.evaluate(validation_generator)
    print(f'\nEvaluation Loss: {evaluation[0]}')
    print(f'Evaluation Accuracy: {evaluation[1]}')

    # Generate predictions
    y_pred = model.predict(validation_generator)
    y_pred_classes = tf.argmax(y_pred, axis=1)
    true_labels = validation_generator.classes

    # Classification Report
    print("\nClassification Report:\n", classification_report(true_labels, y_pred_classes))

    # Confusion Matrix
    conf_matrix = confusion_matrix(true_labels, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=train_generator.class_indices,
                yticklabels=train_generator.class_indices)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()

    # Save the model
    model.save('multi_model_classifier.h5')

    # Clear the session
    tf.keras.backend.clear_session()


# Set your dataset path
dataset_path = "C:\\Users\\ELCOT\\OneDrive\\Documents\\Pills dataset"

# Run the training and evaluation
train_and_evaluate(dataset_path)
