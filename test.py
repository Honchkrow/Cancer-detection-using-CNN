import numpy as np
from keras.layers import Dense, Flatten, Conv2D
from keras.layers import MaxPooling2D, Activation
from keras.models import Sequential
from keras.models import load_model

import keras.backend as K
import tensorflow as tf
import keras


def create_model():
    model = Sequential()

    model.add(Conv2D(16, 3, 3, input_shape=(512, 512, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(32, 3, 3))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, 3, 3))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, 3, 3))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(32, 3, 3))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation("relu"))

    model.add(Dense(128))
    model.add(Activation("relu"))

    model.add(Dense(4))
    model.add(Activation("softmax"))

    model.summary()

    return model


def train(batch_size, epochs):
    config = tf.ConfigProto()
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    model = create_model()
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])

    while True:
        try:
            model = load_model(modelSavePath)
        except:
            print("Training a new model")

        model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)

        model.save(modelSavePath)

        # 注意evaluate和下面的predict的区别
        loss, accuracy = model.evaluate(
            X_test, Y_test_orig, batch_size=1, verbose=1, sample_weight=None
        )

        print("Loss = " + str(loss))
        print("Test Accuracy = " + str(accuracy) + "\n\n\n")
        ch = input("Do you wish to continue training? (y/n) ")
        if ch == "y":
            epochs = int(input("How many epochs this time? : "))
            continue
        else:
            break

    return model


################## Program Start ##################

K.set_image_data_format("channels_last")

batchSize = 64
numOfEpoches = 5
classes = []
modelSavePath = "Breast_Cancer.h5"

print("1. Do you want to train the network\n" "2. Test the model\n(Enter 1 or 2)?\n")
ch = int(input())
if ch == 1:
    classes = np.load("classes.npy")
    print("Loading")
    X_train = np.load("X_train.npy")
    Y_train = np.load("Y_train.npy")
    X_test = np.load("X_test.npy")
    Y_test_orig = np.load("Y_test_orig.npy")

    print("number of training examples = " + str(X_train.shape[0]))
    print("number of test examples = " + str(X_test.shape[0]))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(Y_test_orig.shape))
    model = train(batch_size=batchSize, epochs=numOfEpoches)

elif ch == 2:  # 这里更多是说明来了一幅新图片之后如何进行预测
    # 模拟来了一张新的图片
    X_test = np.load("X_test.npy")
    Y_test_orig = np.load("Y_test_orig.npy")

    ranNum = np.random.randint(0, X_test.shape[0] - 1)
    testImgsX = X_test[ranNum]
    testImgsY = Y_test_orig[ranNum]

    model = load_model(modelSavePath)
    testImgsX = np.expand_dims(testImgsX, axis=0)

    softMaxPred = model.predict(testImgsX)
    print("prediction from CNN: " + str(softMaxPred) + "\n")

    print(123)

else:
    print("Please enter only 1 or 2")
