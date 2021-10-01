import cv2
import numpy as np
import dlib
import os

def get_xy(match, center):
    rad = (match.right() - match.left()) // 2
    (x, y) = center
    sx = x - rad
    ex = x + rad
    sy = match.top()
    ey = match.bottom()
    return sx, ex, sy, ey


def get_aligned_img(datasets, outputs):
    find = dlib.get_frontal_face_detector()
    change = dlib.shape_predictor('./dataset/shape_predictor_68_face_landmarks.dat')

    for (i, dataset) in enumerate(datasets):
        output = outputs[i]
        cnt = len(os.listdir(dataset))

        for j in range(cnt):
            image_name = dataset + str(j + 1) + '.jpg'
            img = cv2.imread(image_name)

            height = img.shape[0]
            width = img.shape[1]
            one = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            catches = find(one, 1)

            for (k, catch) in enumerate(catches):
                points = np.matrix([[t.x, t.y] for t in change(one, catch).parts()])
                left = np.mean(points[42:48], axis=0).astype("int")
                right = np.mean(points[36:42], axis=0).astype("int")

                dx = right[0, 0] - left[0, 0]
                dy = right[0, 1] - left[0, 1]
                d = np.degrees(np.arctan2(dy, dx)) - 180
                length = np.sqrt((pow(dx, 2)) + pow(dy, 2))
                ratio = (left[0, 0] - right[0, 0]) / length
                cx = (right[0, 0] + left[0, 0]) // 2
                cy = (right[0, 1] + left[0, 1]) // 2

                get = cv2.getRotationMatrix2D((int(cx), int(cy)), d, ratio)
                align = cv2.warpAffine(img, get, (width, height), flags=cv2.INTER_CUBIC)

                (sx, ex, sy, ey) = get_xy(catch, (cx, cy))

                cut = align[sy:ey, sx:ex]
                aligned_file = cv2.resize(cut, (200, 200))
                file_path = output + str(j + 1) + ".jpg"
                cv2.imwrite(file_path, aligned_file)

    cv2.waitKey(0)
    cv2.destroyAllWindows()