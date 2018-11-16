from keras.datasets import mnist
from keras.datasets import cifar10
from keras.datasets import fashion_mnist
from autokeras import ImageClassifier
import tensorflow
import numpy as np
import argparse
import pickle as pickle
import numpy as np
import tensorflow as tf
import os, sys, tarfile, urllib
import scipy as sp
import scipy.io as sio
from scipy.misc import *
import urllib.request
import os
import glob
import pickle
from PIL import Image
import random


# Below lies the code to read in the STL10 dataset and it is taken in parts from,
# https://github.com/ltoscano/STL10/blob/master/stl10_input.py
def read_labels(path_to_labels):
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        labels = labels.astype(np.uint8)
        return labels


def read_all_images(path_to_data):
    with open(path_to_data, 'rb') as f:
        everything = np.fromfile(f, dtype=np.uint8)
        images = np.reshape(everything, (-1, 3, 96, 96))
        images = np.transpose(images, (0, 3, 2, 1))
        return images


def download_and_extract(DATA_DIR):
    DATA_URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'
    dest_directory = DATA_DIR
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        # def _progress(count, block_size, total_size):
        #     sys.stdout.write('\rDownloading %s %.2f%%' % (filename,
        #         float(count * block_size) / float(total_size) * 100.0))
        #     sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath)
        print('Downloaded', filename)
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def read_stl10_data(data_path):
    # path to the binary train file with image data
    train_img_path = os.path.join(data_path,'stl10_binary','train_X.bin')

    # path to the binary train file with labels
    train_label_path = os.path.join(data_path,'stl10_binary','train_y.bin')

    # path to the binary test file with image data
    test_img_path = os.path.join(data_path,'stl10_binary','test_X.bin')

    # path to the binary test file with labels
    test_label_path = os.path.join(data_path,'stl10_binary','test_y.bin')

    download_and_extract(data_path)

    # test to check if the whole dataset is read correctly
    x_train = read_all_images(train_img_path)
    print("Training images",x_train.shape)

    y_train = read_labels(train_label_path)
    print("Training labels",y_train.shape)

    x_test = read_all_images(test_img_path)
    print("Test images",x_test.shape)

    y_test = read_labels(test_label_path)
    print("Test labels",y_test.shape)

    x_train = x_train.astype(np.uint8)
    x_test = x_test.astype(np.uint8)

    return (x_train, y_train), (x_test, y_test)

# STL10 dataset code ends above

# Below lies the code for SVHN
# Parts taken from https://github.com/codemukul95/SVHN-classification-using-Tensorflow/blob/master/load_input.py
def read_svhn_data(data_path):
    train_path = os.path.join(data_path, 'train_32x32')
    train_dict = sio.loadmat(train_path)
    X = np.asarray(train_dict['X'], dtype=np.uint8)

    X_train = []
    for i in range(X.shape[3]):
        X_train.append(X[:,:,:,i])
    X_train = np.asarray(X_train)

    Y_train = train_dict['y']
    for i in range(len(Y_train)):
        if Y_train[i]%10 == 0:
            Y_train[i] = 0

    x_train = X_train
    y_train = np.squeeze(Y_train).astype(np.uint8)

    test_path = os.path.join(data_path, 'test_32x32')
    test_dict = sio.loadmat(test_path)
    X = np.asarray(test_dict['X'], dtype= np.uint8)

    X_test = []
    for i in range(X.shape[3]):
        X_test.append(X[:,:,:,i])
    X_test = np.asarray(X_test)

    Y_test = test_dict['y']
    for i in range(len(Y_test)):
        if Y_test[i]%10 == 0:
            Y_test[i] = 0

    x_test = X_test
    y_test = np.squeeze(Y_test).astype(np.uint8)

    return (x_train, y_train), (x_test, y_test)

def read_devanagari_data(dataset_name, data_directory, class_map, segments=1, directories_as_labels=True, files='**/*.png'):
    # Create a dataset of file path and class tuples for each file
    filenames = glob.glob(os.path.join(data_directory, files))
    classes = (os.path.basename(os.path.dirname(name)) for name in filenames) if directories_as_labels else [None] * len(filenames)
    dataset = list(zip(filenames, classes))
    num_examples = len(filenames)
    print("Number of examples",num_examples)
    image_set = []
    label_set = []
    for index, sample in enumerate(dataset):
        file_path, label = sample
        image = Image.open(file_path)
        image_raw = np.array(image)
        image_raw = image_raw.reshape(32,32,1)
        image_set.append(image_raw)
        label_set.append(class_map[label])
    image_set = np.asarray(image_set, dtype=np.uint8)
    label_set = np.asarray(label_set, dtype=np.uint8)
    print("Done", dataset_name)
    return image_set, label_set


def process_devanagari_directory(data_directory:str):
    data_dir = os.path.expanduser(data_directory)
    train_data_dir = os.path.join(data_dir, 'Train')
    test_data_dir = os.path.join(data_dir, 'Test')

    class_names = os.listdir(train_data_dir) # Get names of classes
    class_name2id = { label: index for index, label in enumerate(class_names) } # Map class names to integer labels

    # Persist this mapping so it can be loaded when training for decoding
    with open(os.path.join(data_directory, 'class_name2id.p'), 'wb') as p:
        pickle.dump(class_name2id, p, protocol=pickle.HIGHEST_PROTOCOL)
    
    x_train, y_train = read_devanagari_data('train', train_data_dir, class_name2id)
    combined = list(zip(x_train, y_train))
    random.shuffle(combined)
    x_train, y_train = zip(*combined)
    x_train = np.asarray(x_train, dtype=np.uint8)
    y_train = np.asarray(y_train, dtype=np.uint8)
    
    x_test, y_test = read_devanagari_data('test', test_data_dir, class_name2id)
    combined = list(zip(x_test, y_test))
    random.shuffle(combined)
    x_test, y_test = zip(*combined)
    x_test = np.asarray(x_test, dtype=np.uint8)
    y_test = np.asarray(y_test, dtype=np.uint8)

    print(x_train.shape, y_train.shape,x_train.dtype, y_train.dtype)
    print(x_test.shape, y_test.shape,x_test.dtype, y_test.dtype)
    
    return (x_train, y_train), (x_test, y_test)

if __name__ == '__main__':
    print(tensorflow.__version__)
    parser = argparse.ArgumentParser(description='Autokeras demo')
    parser.add_argument('--dataset', default='mnist', help='Dataset to run algorithm on')
    parser.add_argument('--datapath', help='Dataset path')
    args = parser.parse_args()
    path_provided = True
    if args.datapath == 'None':
       path_provided = False
    if args.dataset == 'mnist':
       print("Using MNIST")
       (x_train, y_train), (x_test, y_test) = mnist.load_data()
       x_train = x_train.reshape(x_train.shape+(1,))
       x_test = x_test.reshape(x_test.shape+(1,))
    elif args.dataset == 'fashion':
       print("Using Fashion")
       (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
       x_train = x_train.reshape(x_train.shape+(1,))
       x_test = x_test.reshape(x_test.shape+(1,))
    elif args.dataset == 'cifar10':
       print("Using Cifar10")
       (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    elif args.dataset == 'devanagari' and path_provided:
       print("Using Devanagari")
       (x_train, y_train), (x_test, y_test) = process_devanagari_directory(args.datapath)
    elif args.dataset == 'svhn' and path_provided:
       print("Using SVHN")
       (x_train, y_train), (x_test, y_test) = read_svhn_data(args.datapath)
    elif args.dataset == 'stl10' and path_provided:
       print("Using STL10")
       (x_train, y_train), (x_test, y_test) = read_stl10_data(args.datapath)
    #x_train = x_train.reshape(x_train.shape+(1,))
    #x_test = x_test.reshape(x_test.shape+(1,))
    print("Shapes",np.shape(x_train),np.shape(x_test),np.shape(y_train),np.shape(y_test))
    clf = ImageClassifier(verbose=True, augment=False)
    clf.fit(x_train, y_train, time_limit=12 * 60 * 60)
    clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
    y = clf.evaluate(x_test, y_test)
    print(y * 100)
