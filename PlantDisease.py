from tensorflow.keras.utils import load_img, img_to_array
from skimage.segmentation import mark_boundaries, slic, quickshift, watershed, felzenszwalb
import numpy as np
import eli5
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import tensorflow as tf
from lime import lime_image
import lime
import time
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.utils import class_weight
from tensorflow.keras.optimizers import Adam
from keras.layers import Input
from keras_tqdm import TQDMNotebookCallback
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50, VGG16
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization, GlobalAveragePooling2D
from keras.models import Sequential
from sklearn.utils import resample, shuffle
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from IPython.display import display  # Allows the use of display() for DataFrames
from time import time
import matplotlib.pyplot as plt
import seaborn as sns  # Plotting library
import keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.utils import np_utils
from sklearn.datasets import load_files
from tqdm import tqdm
from collections import Counter


# print(os.listdir("/home/hossein/Plant3/train/"))

# no stratified
# data_train_path= '/home/hossein/Citrus/Train'
# data_valid_path = '/home/hossein/Citrus/Valid'
# data_test_path =  '/home/hossein/Citrus/Test'

#  80 - 20 stratified
data_train_path = '/home/hossein/Citrus2/Train'
data_valid_path = '/home/hossein/Citrus2/Valid'
data_test_path = '/home/hossein/Citrus2/Test'

# 80 20 stratified 5 classes
# data_train_path = '/home/hossein/Leaves/Train/'
# data_valid_path = '/home/hossein/Leaves/Valid/'
# data_test_path =  '/home/hossein/Leaves/Test/'


# define function to load train, test, and validation datasets
def load_data_raw(path):
    data = load_files(path)
    files = np.array(data['filenames'])
    targets = np_utils.to_categorical(np.array(data['target']), 4)

    return files, targets


train_filenames, train_targets = load_data_raw(data_train_path)


filenames_trimmed = [filename.split('/')[-2] for filename in train_filenames]
classes_count = Counter(filenames_trimmed)

# Plot the classes
plt.bar(classes_count.keys(), classes_count.values())
plt.xticks()


##### UPSAMPLING ##############

def plot_n_samples(filenames):
    filenames_trimmed = [filename.split('/')[-2] for filename in filenames]
    classes_count = Counter(filenames_trimmed)

    # Plot the classes
    plt.bar(classes_count.keys(), classes_count.values())


# Choose one of the 3 for the feature_name
feature_names = {0: 'Black spot', 1: 'Canker', 2: 'Greening', 3: 'Healthy'}


def upsample(filenames, targets, feature_name, n_samples=164):
    upsample_idx = []

    # Find all the indices for nevus
    for i, path in enumerate(filenames):
        # If feature matches, save the index
        if feature_name in path.split('/'):
            upsample_idx.append(i)

    # Remove selected features from filenames to add the upsampled after
    new_filenames = [filename for i, filename in enumerate(
        filenames) if i not in upsample_idx]
    new_targets = [target for i, target in enumerate(
        targets) if i not in upsample_idx]

    # Upsample
    resampled_x, resampled_y = resample(
        filenames[upsample_idx], targets[upsample_idx], n_samples=n_samples, random_state=0)

    # Add the upsampled features to new_filenames and new_targets
    new_filenames += list(resampled_x)
    new_targets += list(resampled_y)

    return np.array(new_filenames), np.array(new_targets)


# We upsample twice: once for each feature we want upsampled
upsample_train_x, upsample_train_y = upsample(
    train_filenames, train_targets, feature_names[0])
upsample_train_x, upsample_train_y = upsample(
    upsample_train_x, upsample_train_y, feature_names[1])
upsample_train_x, upsample_train_y = upsample(
    upsample_train_x, upsample_train_y, feature_names[3])
plot_n_samples(upsample_train_x)

##########################################################


# Convert the image paths to tensors Manually


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(256, 256))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path)
                       for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


train_filenames = paths_to_tensor(upsample_train_x)
train_targets = upsample_train_y


batch_size = 5
# datagen_train = ImageDataGenerator(rescale=1./255)
# datagen_valid = ImageDataGenerator(rescale=1./255)
# datagen_test = ImageDataGenerator(rescale=1./255)

datagen_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.1,  # randomly shift images horizontally
    height_shift_range=0.1,  # randomly shift images vertically
    horizontal_flip=True)

datagen_valid = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.1,  # randomly shift images horizontally
    height_shift_range=0.1,  # randomly shift images vertically
    horizontal_flip=True)

datagen_test = ImageDataGenerator(
    rescale=1./255)


train_generator = datagen_train.flow(
    train_filenames, train_targets, batch_size=batch_size)
valid_generator = datagen_valid.flow_from_directory(data_valid_path, target_size=(
    256, 256),  batch_size=batch_size, class_mode='categorical', shuffle=False)
test_generator = datagen_test.flow_from_directory(data_test_path, target_size=(
    256, 256),  batch_size=1, class_mode='categorical', shuffle=False)


num_train = len(train_filenames)
num_valid = len(valid_generator.filenames)
num_test = len(test_generator.filenames)
print(num_train, num_valid, num_test)

# Class name to the index
# class_2_indices = train_generator.class_indices
# class_2_indices = {'Apple___Apple_scab': 0, 'Apple___Black_rot': 1, 'Apple___Cedar_apple_rust': 2,  'Apple___healthy': 3, 'Blueberry___healthy': 4, 'Cherry_(including_sour)___healthy': 5, 'Cherry_(including_sour)___Powdery_mildew': 6
#                     ,'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 7,'Corn_(maize)___Common_rust_': 8,'Corn_(maize)___healthy': 9, 'Corn_(maize)___Northern_Leaf_Blight': 10, 'Grape___Black_rot': 11, 'Grape___Esca_(Black_Measles)': 12
#                     , 'Grape___healthy': 13, 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 14, 'Orange___Haunglongbing_(Citrus_greening)': 15, 'Peach___Bacterial_spot': 16, 'Peach___healthy': 17, 'Pepper,_bell___Bacterial_spot': 18, 'Pepper,_bell___healthy': 19
#                     , 'Potato___Early_blight': 20, 'Potato___healthy': 21, 'Potato___Late_blight': 22, 'Raspberry___healthy': 23, 'Soybean___healthy': 24, 'Squash___Powdery_mildew': 25, 'Strawberry___healthy': 26 , 'Strawberry___Leaf_scorch': 27, 'Tomato___Bacterial_spot': 28
#                     , 'Tomato___Early_blight': 29, 'Tomato___healthy': 30, 'Tomato___Late_blight': 31, 'Tomato___Leaf_Mold': 32, 'Tomato___Septoria_leaf_spot': 33, 'Tomato___Spider_mites Two-spotted_spider_mite': 34, 'Tomato___Target_Spot': 35
#                     , 'Tomato___Tomato_mosaic_virus': 36, 'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 37}


class_2_indices = {'Black spot': 0, 'Canker': 1, 'Greening': 2,  'Healthy': 3}

print("Class to index:", class_2_indices)

# Reverse dict with the class index to the class name
indices_2_class = {v: k for k, v in class_2_indices.items()}
print("Index to class:", indices_2_class)


# Lets have a look at some of our images
images, labels = train_generator.next()

fig = plt.figure(figsize=(20, 10))
fig.subplots_adjust(wspace=0.2, hspace=0.4)

# Lets show the first 32 images of a batch
for i, img in enumerate(images[:32]):
    ax = fig.add_subplot(4, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(img)
    image_idx = np.argmax(labels[i])
    ax.set(title=indices_2_class[image_idx])


# Define the model


base_model = ResNet50(weights='imagenet', include_top=False,
                      input_tensor=Input(shape=(256, 256, 3)))

# base_model=ResNet50(weights='imagenet')

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# let's add a fully-connected layer
# x = Dropout(0.8)(x)
# x = Dense(1024, activation='elu')(x)
# x = Dropout(0.95)(x)
# and a logistic layer
predictions = Dense(4, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)


# base_model = VGG16(weights='imagenet')

# # add a global spatial average pooling layer
# x = base_model.output
# # x = GlobalAveragePooling2D()(x)
# # x = MaxAveragePooling2D()(x)
# # let's add a fully-connected layer
# x = Dropout(0.5)(x)
# x = Dense(1024, activation='elu')(x)
# # x = Dropout(0.95)(x)
# # and a logistic layer
# predictions = Dense(4, activation='softmax')(x)

# # this is the model we will train
# model = Model(inputs=base_model.input, outputs=predictions)

# # first: train only the top layers (which were randomly initialized)
# for layer in base_model.layers:
#     layer.trainable = True


# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy',
              metrics=['accuracy'])


# Convert one hot encoded labels to ints
train_targets_classes = [np.argmax(label) for label in train_targets]

# Compute the weights
class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(
                                                      train_targets_classes),
                                                  train_targets_classes)

class_weights_dict = dict(enumerate(class_weights))
print(class_weights_dict)


# train the model
checkpointer = ModelCheckpoint(filepath='/home/hossein/plantUpsampled4.weights.best.hdf5', verbose=1,
                               save_best_only=True)

scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=3, min_lr=1e-8, verbose=1)

early_stopper = EarlyStopping(monitor='val_loss', patience=10,
                              verbose=0, restore_best_weights=True)

history = model.fit_generator(train_generator,
                              # class_weight= class_weights_dict,
                              steps_per_epoch=num_train//batch_size,
                              epochs=30,
                              verbose=0,
                              callbacks=[checkpointer,
                                         scheduler, early_stopper],
                              validation_data=valid_generator,
                              validation_steps=num_valid//batch_size)


############## Model testing###############


# load the weights that yielded the best validation accuracy
# model.load_weights('plantResNet_TopIncludeFalse.weights.best.hdf5')
model.load_weights('/home/hossein/plantUpsampled4.weights.best.hdf5')

yhat = model.predict(X1_test)
score = model.evaluate_generator(test_generator, steps=num_test//1, verbose=1)
print('\n', 'Test accuracy:', score[1])


#########Plot val-loss and train-loss##############

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Training loss', 'Validation loss'], loc='upper right')
plt.show()


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Training accuracy', 'Validation accuracy'], loc='lower right')
plt.show()


# Generate train and test sets for the use of EAI


test_generator.reset()
X1_test, y1_test = next(test_generator)
for i in range(58):  # 1st batch is already fetched before the for loop.
    img, label = next(test_generator)
    X1_test = np.append(X1_test, img, axis=0)
    y1_test = np.append(y1_test, label, axis=0)

    train_generator.reset()
    X1_train, y1_train = next(train_generator)
    for i in range(131):  # 1st batch is already fetched before the for loop.
        img, label = next(train_generator)
        X1_train = np.append(X1_train, img, axis=0)
        y1_train = np.append(y1_train, label, axis=0)


######################LIME############################


start = time.time()
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(
    X1_test[51], model.predict, num_samples=100, segmentation_fn=slic)
image, mask = explanation.get_image_and_mask(model.predict(X1_test[51].reshape(
    (1, 256, 256, 3))).argmax(axis=1)[0], negative_only=False, positive_only=False, num_features=100)
end = time.time()
print("time = ", end-start)
plt.imshow(mark_boundaries((image), mask))
plt.axis('off')


superpixels = quickshift(X1_test[55], n_segments=10, sigma=1)

superpixels = slic(X1_test[55], n_segments=10, sigma=1)
plt.imshow(mark_boundaries(X1_test[55], superpixels))


# Grad-CAM
tf.compat.v1.disable_eager_execution()


# ar_image = (X1_test[14]*255).astype(np.uint8)   # create array
# im = Image.fromarray(np.uint8(ar_image))    # array to image
# im.save("test2.jpg")
# im3 = tf.keras.utils.load_img("/home/hossein/test2.jpg", target_size=model.input_shape[1:3])
# doc=np.array(Image.open("test2.jpg"))
# tensorflow.keras.applications.resnet50.preprocess_input(doc)

start = time.time()
doc = X1_test[53]    # image to array
doc = np.expand_dims(doc, axis=0)
# tf.keras.applications.resnet50.preprocess_input(doc)
predictions = model.predict(doc)
eli5.show_prediction(model, doc)
end = time.time()
print("time = ", end-start)


# eli5.explain_prediction(model,doc)


# import tensorflow as tf
# layer_idx=utils.find_layer_idx(model, 'conv5_block3_out')
# model.layers[-1].activation=tf.keras.activations.linear
# model = utils.apply_modifications(model)

# from tf_keras_vis.gradcam import Gradcam
# cam =gradcam(0,X1_test[1],penultimate_layer=-1)


# model.layers[layer_idx].activation = keras.activations.linear
# model = utils.apply_modifications(model)
# from vis.visualization import visualize_cam


# image = load_img("/home/hossein/cat_dog.jpg")


# for i in range (9):
#     pyplot.subplot(330+1+i)
#     batch=iterator.next()
#     image=batch[0].astype('uint8')
#     pyplot.imshow(image)
# pyplot.show()


# plt.imshow(X1_test[4])
# plt.axis('off')


# creating the dataset
data = {'500': 12, '1000': 24, '2000': 47, '3000': 70, '4000': 96, '5000': 116}
num_samples = list(data.keys())
time = list(data.values())

fig = plt.figure(figsize=(10, 5))

# creating the bar plot
plt.bar(num_samples, time, color='orange',
        width=0.4, label="LIME")
plt.axhline(y=2.36, color='g', linestyle='-', label="Grad-CAM")
plt.xlabel("num_samples")
plt.ylabel("Time(s)")
plt.title("Execution time of LIME Vs. Grad-CAM")
plt.legend(loc='upper left')
plt.show()





# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))
 
# set height of bar
LIME = [12, 25, 50, 74, 96, 122]
GE = [48, 94, 183, 271, 367, 452]
 
# Set position of bar on X axis
br1 = np.arange(len(LIME))
br2 = [x + barWidth for x in br1]
 
# Make the plot
plt.bar(br1, LIME, color ='r', width = barWidth,
        edgecolor ='grey', label ='LIME')
plt.bar(br2, GE, color ='g', width = barWidth,
        edgecolor ='grey', label ='Gradient explainer')

plt.axhline(y=2, color='b', linestyle='-', label="Grad-CAM")
# Adding Xticks
plt.xlabel('N', fontweight ='bold', fontsize = 15)
plt.ylabel('Time(s) ', fontweight ='bold', fontsize = 15)
plt.xticks([r + barWidth for r in range(len(LIME))],
        ['500', '1000', '2000', '3000', '4000', '5000'])
 
plt.legend()
plt.show()


from sklearn.metrics import confusion_matrix
confusion_matrix(np.argmax(y1_test, axis=1), np.argmax(yhat, axis=1))
from sklearn.metrics import precision_score, recall_score

print("Precision Score : ",precision_score(np.argmax(y1_test, axis=1), np.argmax(yhat, axis=1), pos_label='positive', average='macro'))
print("Recall Score : ",recall_score(np.argmax(y1_test, axis=1), np.argmax(yhat, axis=1), pos_label='positive',average='macro'))
from sklearn.metrics import f1_score
f1_score(np.argmax(y1_test, axis=1), np.argmax(yhat, axis=1), pos_label='positive',average='macro')
