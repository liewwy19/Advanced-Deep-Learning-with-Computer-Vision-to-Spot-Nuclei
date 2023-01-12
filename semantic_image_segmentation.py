"""
Created on Thu Jan 12 09:08:12 2023

@author: Wai Yip LIEW (liewwy19@gmail.com)
"""
# %%
#   1. Import packages
import numpy as np
import tensorflow as tf
import os, cv2, warnings
from tensorflow import keras
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tensorflow_examples.models.pix2pix import pix2pix

# surpress tensorflow warning messages 
warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#   1.1 constant variable
_SEED = 142857
EPOCHS = 20
BATCH_SIZE = 16
BUFFER_SIZE = 1000
IMG_SIZE = (224,224)
SAVED_MODEL_PATH = os.path.join(os.getcwd(),'saved_models')
DATA_PATH = os.path.join(os.getcwd(),'datasets','data-science-bowl-2018-2')

#   1.2 define class and functions
class Augment(keras.layers.Layer):
    """Define data augmentation pipeline as a single layer through subclassing (using inheriting)"""
    def __init__(self, seed=_SEED):
        super().__init__()
        self.seed = seed
        self.augment_inputs = tf.keras.layers.RandomRotation(0.3,seed=self.seed)
        self.augment_labels = tf.keras.layers.RandomRotation(0.3,seed=self.seed)
    
    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels


class DisplayCallback(tf.keras.callbacks.Callback):
    """Define a callbacks function to show sample prediction after each epoch (using inheriting)"""
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\nSample prediction after epoch {}\n'.format(epoch+1))
        # return super().on_epoch_end(epoch, logs)


def create_mask(pred_mask):
    """This function take in the raw prediction results and return a proper formated mask"""
    pred_mask = tf.argmax(pred_mask,axis=-1)
    pred_mask = pred_mask[...,tf.newaxis]
    return pred_mask[0]


def display(display_list):
    """This function display the images from the display list"""
    plt.figure(figsize=(15,15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def load_image(image_file, dataset, data_path=DATA_PATH):
    """This function handle the image and mask loading"""
    file_name = image_file.split('.')[0]
    img = cv2.imread(os.path.join(data_path,dataset,'inputs',image_file)) 
    mask = cv2.imread(os.path.join(data_path,dataset,'masks',image_file),cv2.IMREAD_GRAYSCALE)
    img,mask = normalize(img,mask)
    return img,mask


def normalize(input_image, input_mask, img_size=IMG_SIZE):
    """This function normalize and resize both the input image and input mask then return both"""
    img = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
    img = cv2.resize(input_image,img_size)/255.0
    mask = np.round(cv2.resize(input_mask,img_size)/255.0).astype(np.int64)
    return img,mask


def show_predictions(dataset=None, num=1):
    """This function format and display input image together with its actual masks and its predicted masks"""
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask,
        create_mask(model.predict(sample_image[tf.newaxis, ...]))])

# %%
#   2. Data Loading
#   2.1 load train dataset 
X_train = []
y_train = []

print('[INFO] Loading TRAINING DATA ...')
for image_file in os.listdir(os.path.join(DATA_PATH,'train','inputs')):
    print('.',end=' ')
    # prevent accidentally read of "desktop.ini" file
    if image_file.split(".")[-1] != 'png': 
        print(f"{image_file} skipped!")
        continue
    img,mask = load_image(image_file,'train')
    X_train.append(img)
    y_train.append(mask)

print('\nTraining Images Loaded:',len(X_train))
print('Training Masks Loaded:',len(y_train))

#   2.2 load test dataset
X_test = []
y_test = []

print('\n[INFO] Loading TESTING DATA ...')
for image_file in os.listdir(os.path.join(DATA_PATH,'test','inputs')):
    print('.',end=' ')
    # prevent accidentally read of "desktop.ini" file
    if image_file.split(".")[-1] != 'png': 
        print(f"{image_file} skipped!")
        continue
    img,mask = load_image(image_file,'test')
    X_test.append(img)
    y_test.append(mask)

print('\nTesting Images Loaded:',len(X_test))
print('Testing Masks Loaded:',len(y_test))
      

# %%
#   3. Data Preprocessing
#   3.1 convert the list of np array into a np array
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

#   3.2 expan the mask dimension
y_train = np.expand_dims(y_train,axis=-1)
y_test = np.expand_dims(y_test,axis=-1)

#   3.3 make sure the mask is in [0,1]
print('y_train:',np.unique(y_train[0]))
print('y_test:',np.unique(y_test[0]))

# %%
#   4. Convert the numpy arrays into tensor slices
X_train = tf.data.Dataset.from_tensor_slices(X_train)
X_test = tf.data.Dataset.from_tensor_slices(X_test)
y_train = tf.data.Dataset.from_tensor_slices(y_train)
y_test = tf.data.Dataset.from_tensor_slices(y_test)

# %%
#   5. Combine the images and masks using the zip method
train_dataset = tf.data.Dataset.zip((X_train,y_train))
test_dataset = tf.data.Dataset.zip((X_test,y_test))

# %%
#   6. Build the datasets


train_batches = (
    train_dataset
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

test_batches = test_dataset.batch(BATCH_SIZE)

# %%
#   7. Visialize some pictures as example
for images, masks in train_batches.take(2):
    sample_image, sample_mask = images[0], masks[0]
    display([sample_image, sample_mask])

# %%
#   8. Model development
#   8.1 use a pre-trained model as the feature extrator
INPUT_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=INPUT_SHAPE,include_top=False)
tf.keras.utils.plot_model(base_model,to_file="base_model.png")

#   8.2 list those activation layers as the outputs from the feature extractor 

layer_name = [
    'block_1_expand_relu',          # 64x64
    'block_3_expand_relu',          # 32x32
    'block_6_expand_relu',          # 16x16
    'block_13_expand_relu',         # 8x8
    'block_16_project'              # 4x4
] 

base_model_outputs = [base_model.get_layer(name).output for name in layer_name]

#   8.3 instantiate the feature extrator 
down_stack = tf.keras.Model(inputs=base_model.input,outputs=base_model_outputs)
down_stack.trainable = False

#   8.4 define the upsampling path
up_stack = [
    pix2pix.upsample(512,3), # 4x4 --> 8x8
    pix2pix.upsample(256,3), # 8x8 --> 16x16
    pix2pix.upsample(128,3), # 16x16 --> 32x32
    pix2pix.upsample(64,3),  # 32x32 --> 64x64
]

# %%
#   8.5 construct the entire U-net using functional API
def unet(output_channels:int):
    inputs = tf.keras.layers.Input(shape=INPUT_SHAPE)
    #Downsample through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    #Build the upsampling path and establich the concatenation
    for up,skip in zip(up_stack,skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x,skip])

    #Use a transpose convolutiona layer to perform the last upsamping, this will become the output layer
    last = tf.keras.layers.Conv2DTranspose(filters=output_channels,kernel_size=3,strides=2,padding='same')  # 64x64 --> 128x128      
    outputs = last(x)

    model = tf.keras.Model(inputs=inputs,outputs=outputs)

    return model

#   8.6 Use the function to create the model
OUTPUT_CHANNELS = 3
model = unet(OUTPUT_CHANNELS)
model.summary()
tf.keras.utils.plot_model(model, to_file="final_model.png")

# %%
#   9. Compile the model
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',loss=loss,metrics=['accuracy'])

# %%
#   10. show pre-train prediction
print('[INFO] "Pre-train" prediction')
show_predictions()

# %%
#   11. Model training
history = model.fit(train_batches,validation_data=test_batches,epochs=EPOCHS,callbacks=[DisplayCallback()])

# %%
#   12. Model evaluation
epoch_no = history.epoch
metrics_dict = history.history

plt.figure(figsize=(8,10))
for key, value in metrics_dict.items():
    if 'loss' in key:
        plt.subplot(2,1,1)
        plt.title('Training and Validation Loss')
        plt.ylabel('Loss Value')
    if 'accuracy' in key:
        plt.subplot(2,1,2)
        plt.title('Training and Validation Accuracy')
        plt.ylabel('Accuracy Value')
    
    plt.plot(epoch_no,value,'r' if 'val' in key else 'bo',label=key)
    # plt.ylim([0, 1])
    plt.legend()

plt.show()

# %%
#   13. Model deployment
#   13.1 make some prediction with test dataset
show_predictions(test_batches, 3)

#   13.2 model saving
model.save(os.path.join(SAVED_MODEL_PATH,'model.h5')) # saving the train model
# %%