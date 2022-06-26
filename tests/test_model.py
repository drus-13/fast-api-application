import cv2
import numpy as np
import os
import tensorflow as tf

PATH_TO_MODEL = r'static/model.tflite'
PATH_TO_TEST_IMGS = r'tests/test_imgs/'

model = tf.lite.Interpreter(PATH_TO_MODEL)
model.allocate_tensors()

input_details = model.get_input_details()
output_details = model.get_output_details()

class_mapping = {0: 'Building',
                 1: 'Forest',
                 2: 'Glacier',
                 3: 'Mountain',
                 4: 'Sea',
                 5: 'Street'}


def model_predict(images_arr):
    predictions = [0] * len(images_arr)

    for i, val in enumerate(predictions):
        model.set_tensor(input_details[0]['index'], images_arr[i].reshape((1, 150, 150, 3)))
        model.invoke()
        predictions[i] = model.get_tensor(output_details[0]['index']).reshape((6,))

    prediction_probabilities = np.array(predictions)
    argmaxs = np.argmax(prediction_probabilities, axis=1)

    return argmaxs


def resize(image):
    return cv2.resize(image, (150, 150))


# Preparing for image inference
def image_processing(images):
    images_resized = [resize(img) for img in images]
    images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images_resized]
    images_arr = np.array(images_rgb, dtype=np.float32)

    return images_arr


def test(path_to_folder):
    for _, _, files in os.walk(path_to_folder):
        path_to_imgs = [path_to_folder + file for file in files]

    images = []
    for path in path_to_imgs:
        img = cv2.imread(path)
        images.append(img)

    images_arr = image_processing(images)
    class_indexes = model_predict(images_arr)

    class_predictions = [class_mapping[x] for x in class_indexes]
    amount_true = 0
    right_answers = [file.split('_')[0].lower() for file in files]
    for id, pred in enumerate(class_predictions):
        if right_answers[id] == pred.lower():
            amount_true += 1
    if len(images) == amount_true:
        print(f'Test was passed. {amount_true}/{len(images)} were predicted right by model {PATH_TO_MODEL}')
    else:
        print(
            f'Test wasnt passed!!! {amount_true}/{len(images)} were predicted right by model {PATH_TO_MODEL}. Check params')


# For test
if __name__ == "__main__":
    test(PATH_TO_TEST_IMGS)
