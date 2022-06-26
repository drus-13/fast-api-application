import tensorflow as tf
import numpy as np
import cv2

PATH_TO_MODEL = r'static/model.tflite'

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


# For test
if __name__ == "__main__":
    # Support multipale files
    path_to_imgs = [r'E:\Deploy ML\fast-api-application\tests\test_imgs\forest_3.jpg']
    images = []
    for path in path_to_imgs:
        img = cv2.imread(path)
        images.append(img)

    images_arr = image_processing(images)
    class_indexes = model_predict(images_arr)

    class_predictions = [class_mapping[x] for x in class_indexes]
    for id, img in enumerate(images):
        cv2.imshow('prediction_' + class_predictions[id] + '_frame_N_' + str(id), img)
        cv2.waitKey()
