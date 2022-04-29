import pandas as pd
import numpy as np
import tensorflow as tf
import warnings

from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from IPython.display import clear_output

warnings.filterwarnings("ignore")

#training = "path to test files goes here ex: C:/Users/user/Desktop/train"
#testing = "path to test files goes here ex: C:/Users/user/Desktop/test"

training = "C:/Users/Pablo/Desktop/programming/334TFProject/train"
testing = "C:/Users/Pablo/Desktop/programming/334TFProject/test"
categories  = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sadness", "Surprise"]

preprocessing = tf.keras.applications.densenet.preprocess_input

imagesTrain = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.05, rescale = 1./255, validation_split = 0.2, preprocessing_function=preprocessing)
imagesTest = ImageDataGenerator(rescale = 1./255, validation_split = 0.2, preprocessing_function=preprocessing)

trainingTheData = imagesTrain.flow_from_directory(directory = training, target_size = (48, 48), batch_size = 64, shuffle  = True, color_mode = "rgb", class_mode = "categorical", subset = "training", seed = 12)

creationOfCharts = imagesTest.flow_from_directory(directory = training, target_size = (48, 48), batch_size = 64, shuffle  = True, color_mode = "rgb", class_mode = "categorical", subset = "validation", seed = 12)

testingTheData = imagesTest.flow_from_directory(directory = testing, target_size = (48, 48), batch_size = 64, shuffle  = False, color_mode = "rgb", class_mode = "categorical", seed = 12)

clear_output()

def feature_extractor(inputs):
    feature_extractor = tf.keras.applications.DenseNet169(input_shape=(48, 48, 3), include_top=False, weights="imagenet")(inputs)
    return feature_extractor

def classifier(inputs):
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer = tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(1024, activation="relu", kernel_regularizer = tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation="relu", kernel_regularizer = tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.Dropout(0.5) (x)
    x = tf.keras.layers.Dense(7, activation="softmax", name="classification")(x)
    return x

def final_model(inputs):
    densenet_feature_extractor = feature_extractor(inputs)
    classOutput = classifier(densenet_feature_extractor)
    return classOutput

def define_compile_model():
    inputs = tf.keras.layers.Input(shape=(48, 48, 3))
    classOutput = final_model(inputs) 
    model = tf.keras.Model(inputs=inputs, outputs = classOutput)
    model.compile(optimizer=tf.keras.optimizers.SGD(0.1), loss="categorical_crossentropy", metrics = ["accuracy"])
    return model

model = define_compile_model()

clear_output()

model.layers[1].trainable = False
model.summary()

earlyStoppingCallback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience= 3, verbose= 1, restore_best_weights=True)

history = model.fit(x = trainingTheData, epochs = 1, validation_data = creationOfCharts, callbacks= [earlyStoppingCallback])
history = pd.DataFrame(history.history)

model.layers[1].trainable = True
model.compile(optimizer=tf.keras.optimizers.SGD(0.001), loss="categorical_crossentropy", metrics = ["accuracy"])

historyNew = model.fit(x = trainingTheData,epochs = 1,validation_data = creationOfCharts)
history = history.append(pd.DataFrame(historyNew.history), ignore_index=True)

model.evaluate(testingTheData)
preds = model.predict(testingTheData)
output1 = np.argmax(preds, axis = 1 )
output2 = np.array(testingTheData.labels)

print(classification_report(output2, output1))
