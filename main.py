#!/usr/bin/python
__author__  = "Lakmal Dharmasena"
__credits__ = "Adithya Selvaprithiviraj"
__version__ = "0.1"
__email__   = "lakmalniranga@gmail.com"
__status__  = "demo"

try:
    from FeatureGen import *
except ImportError:
    print "Make sure FeatureGen.pyc file is in the current directory"
    exit()

try:
    import dlib
    from skimage import io
    import numpy
    import cv2
    import time
    from sklearn.externals import joblib
except ImportError:
    print "Make sure you have OpenCV, dLib, scikit learn and skimage libraries properly installed"
    exit()

emotions    = { 1:"Anger", 2:"Contempt", 3:"Disgust", 4:"Neutral", 5:"Happy", 6:"Sadness", 7:"Surprise" }
font_face   = cv2.FONT_HERSHEY_SIMPLEX  # OpenCV font-face
img2        = cv2.imread('frame.png')   # Video outline frame
cam_port    = 0                         # Camera port ex: 0, 1, 2

def Take_photo():
    count = 20 # Shooting start counter
    video_capture = cv2.VideoCapture(cam_port) # start camera

    while True:
        ret, img = video_capture.read() # get camera frames
        dst = cv2.addWeighted(img, 0.7, img2, 1,0) # mearge frame and video

        # height, width, channels = img.shape
        # img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_CUBIC)

        dets = detector(img, 1) # detect face

        if not dets:
            count = 20 # reset counter if face not found
            cv2.putText(dst, "smile ;)", (20,440), font_face, 1, (0,0,0), 1) # text, before detect face
        if dets:
            count -= 1 # decrese counter, when face detected
            cv2.putText(dst, "Smile! Shoting.. " + str(count),(20,440), font_face, 1,(0,0,0),1) # text, when detecting face

        # looping trough details of faces
        for k,d in enumerate(dets):
            shape = predictor(img, d) # get landmarks of face

            for i in range(1,68): #There are 68 landmark points on each face
                cv2.circle(dst, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=1) # draw landmarks on video

            # after shooting finish, save image
            if count == 0:
                cv2.imwrite("temp.jpg", img)
                video_capture.release() # release camera resuore
                break # break loop

        # set full-screen window
        cv2.namedWindow("Output", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        cv2.imshow("Output", dst) # show video window

        # after shooting finish save image
        if count == 0:
            Predict_Emotion()
            cv2.destroyAllWindows()
            break

        if cv2.waitKey(1) & 0xFF == ord('q'): # press 'q' to kill proccess
            break

def Predict_Emotion():
    shoted_img = cv2.imread("temp.jpg") # get shoted image

    dets = detector(shoted_img, 1) # detect face

    # looping trough details of faces
    for k,d in enumerate(dets):
        shape = predictor(shoted_img, d) # get landmarks of face

        landmarks=[]

        # 68 landmarks in this algorithm
        for i in range(68):
            landmarks.append(shape.part(i).x)
            landmarks.append(shape.part(i).y)

        landmarks=numpy.array(landmarks) # create  numpy array

        # Generating features
        features=generateFeatures(landmarks)
        features= numpy.asarray(features)

        # Performing PCA Transform
        pca_features=pca.transform(features)

        # Predicting using trained model
        emo_predicts=classify.predict(pca_features)

    dst = cv2.addWeighted(shoted_img, 0.7, img2, 1,0) # bind shoted photo and photo frame

    emotion_str = str(emotions[int(emo_predicts[0])]) # detected emotion
    
    # Anger
    if int(emo_predicts[0]) == 1:
        cv2.putText(dst, emotion_str, (20,430), font_face, 1,(0,0,0),1)

    # Contempt
    elif int(emo_predicts[0]) == 2:
        cv2.putText(dst, emotion_str, (20,430), font_face, 1,(0,0,0),1)

    # Disgust
    elif int(emo_predicts[0]) == 3:
        cv2.putText(dst, emotion_str, (20,430), font_face, 1,(0,0,0),1)

    # Neutral
    elif int(emo_predicts[0]) == 4:
        cv2.putText(dst, emotion_str, (20,430), font_face, 1,(0,0,0),1)

    # Happy
    elif int(emo_predicts[0]) == 5:
        cv2.putText(dst, "Thank you for your support <3" ,(20,430), font_face, 1,(0,0,0),1)

    # Sadness
    elif int(emo_predicts[0]) == 6:
        cv2.putText(dst, emotion_str, (20,430), font_face, 1,(0,0,0),1)

    # Surprise
    elif int(emo_predicts[0]) == 5:
        cv2.putText(dst, emotion_str, (20,430), font_face, 1,(0,0,0),1)

    # can't identify
    else:
        cv2.putText(dst, "Try Again :)",(20,430), font_face, 1,(0,0,0),1)


    cv2.namedWindow("Output", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Output", dst)
    cv2.waitKey(delay=5000) # set thank you screen intervel in mili-seconds

    Take_photo() # call again realtime detetcion method


if __name__ == "__main__":
    landmark_path="shape_predictor_68_face_landmarks.dat" # get shape predictor file

    print "Initializing Dlib face Detector.."
    detector= dlib.get_frontal_face_detector()

    print "Loading landmark identification data..."
    try:
        predictor= dlib.shape_predictor(landmark_path)
    except:
        print "Unable to find trained facial shape predictor. \nYou can download a trained facial shape predictor from: \nhttp://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2"
        exit()

    print "Loading trained data....."

    try:
        classify=joblib.load("traindata.pkl")
        pca=joblib.load("pcadata.pkl")
    except:
        print "Unable to load trained data. \nMake sure that traindata.pkl and pcadata.pkl are in the current directory"
        exit()

    Take_photo()
