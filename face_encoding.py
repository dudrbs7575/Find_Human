import cv2
import face_recognition
import pickle
import os

def makeFile(CompleteEncodings, CompleteNames,encoding_file):
    # facial encoding을 disk에 저장.
    data = {"encodings": CompleteEncodings, "names": CompleteNames}
    f = open(encoding_file, "wb")
    f.write(pickle.dumps(data))
    f.close()

# 준비된 dataset의 반복문
def startEncoding(dataset_paths, names):
    image_type = '.jpg'

    encoding_file = './pickle/tmp.pickle'
    model_choice = 'cnn'
    # cnn model이 slow하지만 accuracy가 높아 선택.

    # Encoding된 jpg 파일을 담을 배열
    CompleteEncodings = []
    CompleteNames = []

    for (i, dataset) in enumerate(dataset_paths):
        # Dataset에 존재하는 사람의 name을 추출
        person_name = names[i]

        image_order = len(os.listdir(dataset))  # 디렉토리 내 파일 개수 카운트

        for index in range(image_order):
            img_name = dataset + str(index + 1) + image_type

            # jpg file을 read한 후 openCV에서의 BGR을 rgb로 변경.
            img = cv2.imread(img_name)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # rgb와 cnn을 사용하여 x, y좌표를 획득한 후 face square_box생성.
            square = face_recognition.face_locations(rgb,
                                                     model=model_choice)

            # face를 encoding
            After_encoding = face_recognition.face_encodings(rgb, square)

            for encoding in After_encoding:
                # encoding된 파일을 배열에 append
                print(img_name, person_name, encoding)
                CompleteEncodings.append(encoding)
                CompleteNames.append(person_name)
    makeFile(CompleteEncodings, CompleteNames,encoding_file)