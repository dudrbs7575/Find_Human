import face_recognition
import cv2
import pickle
import time
import os
import dlib

from face_encoding import *
from face_alignment import *

non_target = 'UNKNOWN'

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('ERROR : ' +  directory)

def matching(data, face_encode) :
    face_names = []
    print(data)
    for i in face_encode :
        name = non_target
        face_matches = face_recognition.compare_faces(data["encodings"], i)
        print(face_matches)

        if True in face_matches :
            vote = {} # dictionary for matched faces
            for_matching = [] # for_matched faces
            for (j, flag) in enumerate(face_matches):
                if flag : for_matching.append(j) # if face matched
            
            for j in for_matching :
                name = data["names"][j] # matched name
                vote[name] = vote.get(name, 0) + 1 # vote for matched name
            
            name = max(vote, key = vote.get) # select most matched name
        face_names.append(name) # append the name

    return face_names

def Find_Faces(frame, data):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR -> RGB

    face_location = face_recognition.face_locations(rgb, model = 'hog') # find (x,y) for every face
    # print(face_location)
    face_encode = face_recognition.face_encodings(rgb, face_location) # encoding the find faces
    # print(face_encode)
    # print('face_encode')

    face_names = matching(data, face_encode)
    
    for((t, r, b, l), name) in zip(face_location, face_names) :
        if t - 20 > 20 : y = t - 20 
        else : y = t + 15
        if name == non_target : color = (0, 0, 255) # RED for non target
        else : color = (0, 255, 0) # GREEN
        width = 3

        cv2.rectangle(frame, (l, t), (r, b), color, width) # draw square
        # mid = (r + l) // 2 # value of mid
        cv2.putText(frame, name, (l, y), cv2.FONT_HERSHEY_COMPLEX, 1, color, width) # put text

    frame = cv2.resize(frame, None, fx = 0.5, fy = 0.5) # frame resize
    return frame


if __name__ == "__main__" :
    # input
    targets = ['robert'] 

    image_paths = []
    dataset_paths = []
    for target in targets :
        # createFolder
        createFolder('./dataset/' + str(target))
        # createFolder('./output/' + str(target))
        # createFolder('./pickle' + str(target)
        
        image_path = './images/' + target + '/'
        dataset_path = './dataset/' + target + '/'
        image_paths.append(image_path)
        dataset_paths.append(dataset_path)
        # image_order = len(os.listdir(image_path))
        # print(image_path, image_order)
    
    get_aligned_img(image_paths, dataset_paths)
    startEncoding(dataset_paths, targets)

    pickle_name = './pickle/tmp.pickle'
    video = './videos/tmp.mp4'
    #video = './videos/' + target + '.mp4' # video
    DATA = pickle.loads(open(pickle_name, "rb").read())

    capture = cv2.VideoCapture(video)
    if not capture.isOpened :
        print("Can't open the Video")
        exit(1)

    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) / 2), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2))
    print(size)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video_write = cv2.VideoWriter('./videos/output.avi', fourcc, 30, size)
    while True:
        tmp, frame = capture.read() # read from opened video
        if not tmp: 
            break
        frame = Find_Faces(frame, DATA)
        cv2.imshow("Face_Recognition_video", frame)
        video_write.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): # Hit Q
            break
    print('Finish')
    capture.release()
    video_write.release()
    cv2.destroyAllWindows()