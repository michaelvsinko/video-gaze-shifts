import cv2
import numpy as np
from tensorflow import keras


def load_vgg16_model(weights_path):
    """
    model of CNN
    :param weights_path: required path to .h5 file
    :return:
    """
    conv_base = keras.applications.VGG16(weights='imagenet',
                                         include_top=False,
                                         input_shape=(84, 84, 3))

    model = keras.models.Sequential()
    model.add(conv_base)
    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.load_weights(weights_path)
    # model.summary()
    return model


def calc_center(eye):
    """
    Using for further cropping eyes from images with faces
    :param eye:
    :return: x, y, of center of eye
    """
    center_x = (eye[0][0] + eye[3][0]) // 2
    center_y = (eye[0][1] + eye[3][1]) // 2
    return int(center_x), int(center_y)


def prepare_input(img):
    """
    prepare data for predict
    :param img:
    :return:
    """

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (84, 84))
    img = img.astype("float32")
    img /= 255.
    img = np.array([img])
    return img


def close_eye_detect(frame, eye_model, marks):
    """

    :param rightEye:
    :param face_utils_shape:
    :param leftEye:
    :param weights_path:
    :param frame:
    :return:
    """

    leftEye = marks[36:42]
    rightEye = marks[42:46]

    left_eye_center = calc_center(leftEye)
    right_eye_center = calc_center(rightEye)

    eye_list = {0: 'open', 1: 'close'}

    right_eye_image = frame[right_eye_center[1] - 42:right_eye_center[1] + 42,
                            right_eye_center[0] - 42:right_eye_center[0] + 42]

    right_eye_image = prepare_input(right_eye_image)
    right_prediction = eye_model.predict(right_eye_image)

    '''    left_eye_image = frame[left_eye_center[1] - 42:left_eye_center[1] + 42,
                     left_eye_center[0] - 42:left_eye_center[0] + 42]
    left_eye_image = prepare_input(left_eye_image)
    left_eye_image = eye_model.predict(left_eye_image)'''


    right_prediction = float(right_prediction)
    right_prediction = 0 if right_prediction > 0.15 else 1

    return right_prediction


def blinks_count(smiles_list):
    """
    groups
    :param smiles_list:
    :return:
    """
    res = []
    for i in range(0, len(smiles_list), 4):
        a = smiles_list[i: i + 4]
        res.append(1) if sum(a) >= 2 else res.append(0)

    for element in range(1, len(res) - 1):
        if res[element - 1] != res[element] and \
                res[element] != res[element + 1]:
            res[element] = res[element - 1]
    change_blinks = []  #
    for k in res:
        if len(change_blinks) == 0:
            change_blinks.append(k)
        else:
            if k != change_blinks[-1]:
                change_blinks.append(k)
    return sum(change_blinks)
