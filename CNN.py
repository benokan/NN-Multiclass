from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
import numpy as np
from glob import glob
from PIL import ImageFile
import random
from tqdm import tqdm
from keras.preprocessing import image
from keras.utils import np_utils


random.seed(340958234985)
np.random.seed(2093846)

image_names = [item for item in glob("training/sc5/*/")]
number_of_image_categories = len(image_names)
print("-- Training Data Information --")
print('%d image categories.' % number_of_image_categories)
print(image_names)

test_data = [ items for items in glob('test/*.jpg')]
print("%d test images found" %len(test_data))

def load_dataset(path):
    data = load_files(path)
    image_files = np.array(data['filenames'])
    image_targets = np_utils.to_categorical(np.array(data['target']), number_of_image_categories)
    return image_files, image_targets


image_files, image_targets = load_dataset('training/sc5/')


trains_validate_files, test_files, trains_validate_targets, test_targets = \
    train_test_split(image_files, image_targets, test_size=0.2, random_state=42)

train_files, valid_files, train_targets, valid_targets = \
    train_test_split(trains_validate_files, trains_validate_targets, test_size=0.25, random_state=42)




image_names = [item[20:-1] for item in sorted(glob("training/sc5/*/"))]


print('%d training images.' % len(train_files))
print('%d validation images.' % len(valid_files))
print('%d test images.' % len(test_files))


def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(100, 100))
    img_array = image.img_to_array(img)

    return np.expand_dims(img_array, axis=0)


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


ImageFile.LOAD_TRUNCATED_IMAGES = True

train_tensors = paths_to_tensor(train_files).astype('float32') / 255
valid_tensors = paths_to_tensor(valid_files).astype('float32') / 255
test_tensors = paths_to_tensor(test_files).astype('float32') / 255

# vgg = keras.applications.vgg16.VGG16(weights=None,classes=24,input_shape=(100,100,3))
# vgg.compile(optimizer="adam",loss='categorical_crossentropy')
# vgg.fit(train_tensors, train_targets, validation_data=(valid_tensors, valid_targets),
#            epochs=5, batch_size=64, verbose=1)
#
#
#
#
#




# model = Sequential()
#
# model.add(Conv2D(filters=4, kernel_size=2, padding='same',
#                  activation='relu', input_shape=(100, 100, 3)))
# model.add(MaxPooling2D(pool_size=2))
#
# model.add(Conv2D(filters=8, kernel_size=2, padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.1))
#
# model.add(Conv2D(filters=12, kernel_size=2, padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.2))
#
# model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.3))
#
# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.4))
#
# model.add(Dense(24, activation='softmax'))
#
#
# model.summary()
#
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# epochs = 5
#
# checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.image_classifier.hdf5',
#                                verbose=1, save_best_only=True)
#
# model.fit(train_tensors, train_targets, validation_data=(valid_tensors, valid_targets),
#           epochs=epochs, batch_size=64, callbacks=[checkpointer], verbose=1)
#
# model.load_weights('saved_models/weights.best.image_classifier.hdf5')
#
# predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
#
# test_accuracy = 100*np.sum(np.array(predictions) == np.argmax(test_targets, axis=1))/len(predictions)
# print('Test accuracy: %.4f%%' % test_accuracy)
