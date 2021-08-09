from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import applications
from matplotlib import pyplot as plt
from tensorflow.keras import layers, Sequential
import numpy as np
from tensorflow.keras.models import load_model
import seaborn as sns
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# some parameter
data_dir = './brain-tumor2'
image_size = 150
batch_size = 64
seed = 123456
validation_split = 0.2
# model_name can be 'basic-cnn' 'efficient-net-b0~7' 'vgg16' 'vgg19' 'resnet50' 'resnet101'
model_name = 'basic-cnn'
epochs = 50

# prepare data
train_ds = image_dataset_from_directory(
    data_dir,
    seed=seed,
    validation_split=validation_split,
    subset="training",
    image_size=(image_size, image_size),
    batch_size=batch_size)

val_ds = image_dataset_from_directory(
    data_dir,
    seed=seed,
    validation_split=validation_split,
    subset="validation",
    image_size=(image_size, image_size),
    batch_size=batch_size)

labels = train_ds.class_names
print(f'class name {labels}')

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# create a model
if model_name == 'basic-cnn':
    # add the data augmentation layer to reduce val loss
    data_augmentation = Sequential(
        [
            layers.experimental.preprocessing.RandomFlip(
                "horizontal", input_shape=(image_size, image_size, 3)),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1)
        ]
    )
    model = Sequential([
        data_augmentation,
        layers.experimental.preprocessing.Rescaling(
            1./255, input_shape=(image_size, image_size, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(4, activation='softmax')
    ])
elif model_name.startswith('efficient'):
    i = layers.Input([image_size, image_size, 3], dtype=tf.uint8)
    x = tf.cast(i, tf.float32)
    # core includes preprocess
    if model_name.endswith('b0'):
        core = applications.EfficientNetB0(include_top=False)
    elif model_name.endswith('b1'):
        core = applications.EfficientNetB1(include_top=False)
    elif model_name.endswith('b2'):
        core = applications.EfficientNetB2(include_top=False)
    elif model_name.endswith('b3'):
        core = applications.EfficientNetB3(include_top=False)
    elif model_name.endswith('b4'):
        core = applications.EfficientNetB4(include_top=False)
    elif model_name.endswith('b5'):
        core = applications.EfficientNetB5(include_top=False)
    elif model_name.endswith('b6'):
        core = applications.EfficientNetB6(include_top=False)
    elif model_name.endswith('b7'):
        core = applications.EfficientNetB7(include_top=False)
    x = core(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(4, activation='softmax')(x)
    model = tf.keras.Model(inputs=[i], outputs=[x])
elif model_name.startswith('vgg'):
    i = layers.Input([image_size, image_size, 3], dtype=tf.uint8)
    x = tf.cast(i, tf.float32)
    if model_name.endswith('16'):
        x = applications.vgg16.preprocess_input(x)
        # using imagenet pretrained weight to initialize model often causes
        # the model training process gets stuck
        core = applications.vgg16.VGG16(include_top=False, weights=None)
    elif model_name.endswith('19'):
        x = applications.vgg19.preprocess_input(x)
        core = applications.vgg19.VGG19(include_top=False, weights=None)
    x = core(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4, activation="softmax")(x)
    model = tf.keras.Model(inputs=[i], outputs=[x])
elif model_name.startswith('resnet'):
    i = layers.Input([image_size, image_size, 3], dtype=tf.uint8)
    x = tf.cast(i, tf.float32)
    x = applications.resnet.preprocess_input(x)
    if model_name.endswith('50'):
        core = applications.resnet.ResNet50(include_top=False)
    elif model_name.endswith('101'):
        core = applications.resnet.ResNet101(include_top=False)
    x = core(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4, activation="softmax")(x)
    model = tf.keras.Model(inputs=[i], outputs=[x])
elif model_name.startswith('mobilenet'):
    i = layers.Input([image_size, image_size, 3], dtype=tf.uint8)
    x = tf.cast(i, tf.float32)
    x = applications.mobilenet.preprocess_input(x)
    core = applications.mobilenet.MobileNet(include_top=False)
    x = core(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4, activation="softmax")(x)
    model = tf.keras.Model(inputs=[i], outputs=[x])


model.summary()

model.compile(optimizer='adam', loss='SparseCategoricalCrossentropy',
              metrics=['accuracy'])


tensorboard = TensorBoard(log_dir='logs')
checkpoint = ModelCheckpoint(
    f"{model_name}.h5", monitor="val_accuracy", save_best_only=True, mode="auto", verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=5, min_delta=0.001,
                              mode='auto', verbose=1)

# start training
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[tensorboard, checkpoint, reduce_lr]
)


# plot loss and accuracy curve
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

print('drawing the learning curve')
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig(f'{model_name} learning curve.png')
print('saving the learning curve')

# evaluation
print('evaluating the best model...')
model = load_model(f'{model_name}.h5')
model.evaluate(val_ds)

# print classification report
prediction = []
y = []
for i in val_ds:
    pred = model.predict(i[0])
    pred = np.argmax(pred, axis=1)
    prediction.extend(pred)
    y.extend(i[1])
report = classification_report(y, prediction)
print(report)

# draw confusion matrix
print('drawing confusion matrix')
colors_dark = ["#1F1F1F", "#313131", '#636363', '#AEAEAE', '#DADADA']
colors_red = ["#331313", "#582626", '#9E1717', '#D35151', '#E9B4B4']
colors_green = ['#01411C', '#4B6F44', '#4F7942', '#74C365', '#D0F0C0']

fig, ax = plt.subplots(1, 1, figsize=(14, 7))
sns.heatmap(confusion_matrix(y, prediction), ax=ax, xticklabels=labels, yticklabels=labels, annot=True,
            cmap=colors_green[::-1], alpha=0.7, linewidths=2, linecolor=colors_dark[3])
fig.text(s='Heatmap of the Confusion Matrix', size=18, fontweight='bold',
         fontname='monospace', color=colors_dark[1], y=0.92, x=0.28, alpha=0.8)

plt.savefig(f'{model_name} confusion matrix.png')
print('save confustion matrix')
