from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# initialize the initial learning rate, number of epochs to train for,
learning_rate = 0.001
epochs = 50
batch_size = 32


data = []
labels = []


#convert images to 224x224 and append images to data list (labels correspond to the image)
print("loading images...")
for category in ["with_mask","without_mask"]:
    for face in os.listdir(f"./images/{category}"):
        img_path = f"./images/{category}/{face}"
        img = load_img(img_path, target_size=(224, 224))
        img = img_to_array(img)
        img = preprocess_input(img)
        data.append(img)
        labels.append(category)

""" 
data is a list of images numerical arrays
but labels is list of strings ("with mask" or "without mask") 
Labels to be turned into one-shot arrays
"""

#converts to 0s and 1s
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

#convert the labels and data into numpy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

""" 
Split the data into training and testing sets
20% of the data will be used for testing
80% of the data will be used for training
"""
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=50)


""" 
generate training data. Uses one image to create many images
this is done to create a more diverse dataset
--- Flip image - rotate - zoom etc. ---
"""

augmentation = ImageDataGenerator(
    rotation_range=22,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
    )

#neural network we will use is called MobileNetV2 (pretrained)
"""
For image classification - Im not going to bother explaining this
shape = 244x244 and 3 is RGB (3 channels)
"""
    
print("Creating model...")
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))
#pool size 7,7
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

#similar to in AI, when we call Model, we give an input and recieve an output. 
model = Model(inputs=baseModel.input, outputs=headModel)

#do not train the base model (meaning first model)
for layer in baseModel.layers:layer.trainable = False

opt = Adam(lr=learning_rate, decay=learning_rate / epochs)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

#train the models head
print("Training network")
H = model.fit(
	augmentation.flow(trainX, trainY, batch_size=batch_size),
	steps_per_epoch=len(trainX) // batch_size,
	validation_data=(testX, testY),
	validation_steps=len(testX) // batch_size,
	epochs=epochs) 

#evaluation
print("Evaluating network...")
predX = model.predict(testX, batch_size=batch_size)

"""
find highest probability from data (correspond to labels) 
print the classification report formatted
save the model
"""
predX = np.argmax(predX, axis=1)
print(classification_report(testY.argmax(axis=1), predX, target_names=lb.classes_))

print("Saving model...")
model.save("mask_detector.model", save_format="h5")


# plot the training loss and accuracy
N = epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")