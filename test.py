from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Lambda, Activation, Conv2D
from keras.layers.core import Flatten, Dense
from keras.layers.convolutional import MaxPooling2D
import os.path
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from keras import optimizers

dict_train = {
    'Alilaguna': 0,
    'Ambulanza': 1,
    'Barchino': 2,
    'Cacciapesca': 3,
    'Caorlina': 4,
    'Gondola': 5,
    'Lanciafino10m': 6,
    'Lanciafino10mBianca': 7,
    'Lanciafino10mMarrone': 8,
    'Lanciamaggioredi10mBianca': 9,
    'Lanciamaggioredi10mMarrone': 10,
    'Motobarca': 11,
    'Motopontonerettangolare': 12,
    'MotoscafoACTV': 13,
    'Mototopo': 14,
    'Patanella': 15,
    'Polizia': 16,
    'Raccoltarifiuti': 17,
    'Sandoloaremi': 18,
    'Sanpierota': 19,
    'Topa': 20,
    'VaporettoACTV': 21,
    'VigilidelFuoco': 22,
    'Water': 23

}
dict_test = {
    'Alilaguna': 0,
    'Ambulanza': 1,
    'Barchino': 2,
    'Cacciapesca': 3,
    'Caorlina': 4,
    'Gondola': 5,
    'Lanciafino10m': 6,
    'Lanciafino10mBianca': 7,
    'Lanciafino10mMarrone': 8,
    'Lanciamaggioredi10mBianca': 9,
    'Lanciamaggioredi10mMarrone': 10,
    'Motobarca': 11,
    'Motopontonerettangolare': 12,
    'MotoscafoACTV': 13,
    'Mototopo': 14,
    'Patanella': 15,
    'Polizia': 16,
    'Raccoltarifiuti': 17,
    'Sandoloaremi': 18,
    'Sanpierota': 19,
    'Topa': 20,
    'VaporettoACTV': 21,
    'VigilidelFuoco': 22,
    'Water': 23

}

train_folder = "training/sc5/"
test_folder = "test/"
size_image = 100

def save_data(X, y, name):
    with open(name + '.p', 'wb') as f:
        pickle.dump({'features': X, 'labels': y}, f, pickle.HIGHEST_PROTOCOL)


X_train = []
y_train = []

if os.path.isfile('training/sc5.p'):
    print('Training file found!')
    with open('training/sc5.p', mode='rb') as f:
        train = pickle.load(f)
    X_train, y_train = train['features'], train['labels']
else:
    print('Training file not found!')

    for i in os.listdir(train_folder):
        subfolder = train_folder + i + "/"
        if os.path.isdir(subfolder):
            files = os.listdir(subfolder)

            # If file is in classes to be classified
            if i in dict_train:
                pbar = tqdm(range(len(files)))
                pbar.set_description('Processing \'' + i + '\'')

                # For each file
                for j in pbar:
                    f = files[j]
                    image = cv2.resize(plt.imread(subfolder + f),
                                       (size_image, size_image),
                                       interpolation=cv2.INTER_LINEAR)
                    X_train.append(image)
                    y_train.append(dict_train[i])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Save dictionary
    save_data(X_train, y_train, 'training/sc5')
    print('\nTraining saved to file!')

X_test = []
y_test = []

# load test files
if os.path.isfile('test.p'):
    print('Test file found!')
    with open('test.p', mode='rb') as f:
        test = pickle.load(f)
    X_test, y_test = test['features'], test['labels']
else:
    print('Test file not found!')

    test_label_dictionary = {}
    for l in open('ground_truth.txt', 'r'):
        file, name = l.strip().split(';')
        if name in dict_test:
            test_label_dictionary[file] = name

    pbar = tqdm(range(len(test_label_dictionary)))
    pbar.set_description('Processing test data')

    keys = list(test_label_dictionary)

    for f in pbar:
        image = cv2.resize(plt.imread(test_folder + keys[f]),
                           (size_image, size_image),
                           interpolation=cv2.INTER_LINEAR)
        X_test.append(image)
        y_test.append(dict_test[test_label_dictionary[keys[f]]])

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Save dictionary
    save_data(X_test, y_test, 'test')
    print('\nTest saved to file!')

# Number of training examples
n_train = len(X_train)

# Number of testing examples
n_test = len(X_test)

# Shape of the image
image_shape = X_test[0].shape

# Number of classes
n_classes = len(dict_train)

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


label_binarizer = preprocessing.LabelBinarizer()
label_binarizer.fit(y_train)

# Shuffle the data
X_train, y_train = shuffle(X_train, y_train)

# Encode labels
y_train = label_binarizer.transform(y_train)
y_test = label_binarizer.transform(y_test)


X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.float32)

height, width, depth = X_train[0].shape


def model(width, height, depth, classes, weightsPath=None):
    model = Sequential()

    model.add(Lambda(lambda x: x / 127.5 - 1.,
                     input_shape=(width, height, depth),
                     output_shape=(width, height, depth)))

    model.add(Conv2D(20, (5, 5), padding="same",
                     input_shape=(depth, height, width)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(50, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))

    model.add(Dense(classes))
    model.add(Activation("softmax"))

    # if a weights path is supplied (inicating that the model was
    # pre-trained), then load the weights
    if weightsPath is not None:
        model.load_weights(weightsPath)


    return model


model = model(width, height, depth, n_classes, weightsPath=None)

model.summary()
print("Training model...")
opt = optimizers.Adam(lr=0.001)
model.compile(loss="categorical_crossentropy",
              optimizer=opt, metrics=["accuracy"])

# Train model
model.fit(X_train, y_train, batch_size=20, epochs=10, verbose=2)