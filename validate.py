import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import load_model
import tensorflow as tf

model_names = ['basic-cnn','efficient-net-b0','efficient-net-b1','efficient-net-b2',\
'vgg16','vgg19','resnet50','resnet101']
data_dir = './brain-tumor2'
image_size = 150
batch_size = 64
seed = 123456
validation_split = 0.2

val_ds = image_dataset_from_directory(
  data_dir,
  seed=seed,
  validation_split=validation_split,
  subset="validation",
  image_size=(image_size, image_size),
  batch_size=batch_size)


AUTOTUNE = tf.data.AUTOTUNE

val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

for name in model_names:
    print(f'loading model: {name}.h5 ...')
    model = load_model(f'{name}.h5')
    model.evaluate(val_ds)
    print('----------------------------')