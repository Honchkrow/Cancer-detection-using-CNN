import numpy as np
from keras.layers import Input, Dense, ZeroPadding2D, Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout
from keras.models import Model
from keras.models import load_model

from PIL import Image
import keras.backend as K
import tensorflow as tf
import keras


def defModel(input_shape):
    x = Input((512, 512, 3))
    y = Conv2D(64, (3, 3), activation="relu")(x)
    y = MaxPooling2D(pool_size=(3, 3))(y)
    y = Conv2D(64, (3, 3), activation="relu")(y)
    y = MaxPooling2D(pool_size=(2, 2))(y)
    y = Conv2D(32, (3, 3), activation="relu")(y)
    y = ZeroPadding2D(padding=(1, 1))(y)
    y = MaxPooling2D(pool_size=(3, 3))(y)
    y = Conv2D(16, (3, 3), activation="relu")(y)
    y = MaxPooling2D(pool_size=(2, 2))(y)
    y = Conv2D(16, (3, 3), activation="relu")(y)
    y = Flatten()(y)
    y = Dense(256, activation="relu")(y)
    y = Dense(128, activation="relu")(y)
    y = Dropout(0.5)(y)
    y = Dense(4, activation="softmax")(y)

    # ------------------------------------------------------------------------------

    model = Model(inputs=x, outputs=y, name="Model")
    model.summary()

    return model


def train(batch_size, epochs):
    config = tf.ConfigProto()
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)
    model = defModel(X_train.shape[1:])
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])

    while True:
        try:
            model = load_model(modelSavePath)
        except:
            print("Training a new model")

        model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)

        model.save(modelSavePath)

        preds = model.evaluate(
            X_test, Y_test_orig, batch_size=1, verbose=1, sample_weight=None
        )
        print(preds)

        print()
        print("Loss = " + str(preds[0]))
        print("Test Accuracy = " + str(preds[1]) + "\n\n\n\n\n")
        ch = input("Do you wish to continue training? (y/n) ")
        if ch == "y":
            epochs = int(input("How many epochs this time? : "))
            continue
        else:
            break

    return model


def predict(img, savedModelPath, showImg=True):
    model = load_model(savedModelPath)

    x = img
    if showImg:
        Image.fromarray(np.array(img, np.float16), "RGB").show()
    x = np.expand_dims(x, axis=0)

    softMaxPred = model.predict(x)
    print("prediction from CNN: " + str(softMaxPred) + "\n")
    probs = softmaxToProbs(softMaxPred)
    maxprob = 0
    maxI = 0
    for j in range(len(probs)):
        if probs[j] > maxprob:
            maxprob = probs[j]
            maxI = j
    print("prediction index: " + str(maxI))
    return maxI, probs


def softmaxToProbs(soft):
    z_exp = [np.math.exp(i) for i in soft[0]]
    sum_z_exp = sum(z_exp)
    return [(i / sum_z_exp) * 100 for i in z_exp]


################## Program Start ##################

K.set_image_data_format("channels_last")

modelSavePath = "Breast_Cancer.h5"
numOfTestPoints = 2
batchSize = 64
numOfEpoches = 5
classes = []

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

elif ch == 2:
    classes = np.load("classes.npy")
    print("Loading")
    X_train = np.load("X_train.npy")
    Y_train = np.load("Y_train.npy")
    X_test = np.load("X_test.npy")
    Y_test_orig = np.load("Y_test_orig.npy")

    _ = None
    __ = None
    testImgsX = []
    testImgsY = []
    ran = []
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))
    # print(X_train[1])
    for i in range(10):
        ran.append(np.random.randint(0, X_train.shape[0] - 1))
    for ranNum in ran:
        testImgsX.append(X_train[ranNum])
        testImgsY.append(Y_train[ranNum])

    X_train = None
    Y_train = None

    print("testImgsX shape: " + str(len(testImgsX)))
    print("testImgsY shape: " + str(len(testImgsY)))

    cnt = 0.0

    classes = []
    classes.append("Benign")
    classes.append("InSitu")
    classes.append("Invasive")
    classes.append("Normal")

    compProbs = []
    compProbs.append(0)
    compProbs.append(0)
    compProbs.append(0)
    compProbs.append(0)

    for i in range(len(testImgsX)):
        print("\n\nTest image " + str(i + 1) + " prediction:\n")

        predi, probs = predict(testImgsX[i], modelSavePath, showImg=False)

        for j in range(len(classes)):
            print(str(classes[j]) + " : " + str(round(probs[j], 4)) + "%")
            compProbs[j] += probs[j]

        maxi = 0
        for j in range(len(testImgsY[0])):
            if testImgsY[i][j] == 1:  # The right class
                maxi = j
                break
        if predi == maxi:
            cnt += 1

    print("% of images that are correct: " + str((cnt / len(testImgsX)) * 100))

else:
    print("Please enter only 1 or 2")
