import numpy as np
import cv2
import dlib
from threading import Thread



class WebcamVideoStream: #multiThread video acq
    def __init__(self, src=0, w=320, h=240):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        self.stream.set(3,w)
        self.stream.set(4,h)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while(self.stream.isOpened()):
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

###########MAIN 

vs = WebcamVideoStream(src=0,w=640,h=480).start()

print('cacca')
dlibHogdetector = dlib.get_frontal_face_detector()

print('Loading landmark model')
NLandmark = 5  # or 68
landmarkfn = "shape_predictor_" + str(NLandmark) + "_face_landmarks.dat"
predictor = dlib.shape_predictor('/home/francesco/Seafile/SmartVend/FaceDet/objDetectors/landmarkDet/' + landmarkfn)

# clahe object for contrast limitng histogram equalization
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

print ('culo')
print("Loading CNN")
NFEATURES = 128
DATA_AUGM_N = 1 #only one check
face_rec_model_path_base = "faceRecmodels/"
modelName = "dlib_face_recognition_resnet_model_v1.dat"
facerec = dlib.face_recognition_model_v1('/home/francesco/Seafile/SmartVend/FaceDet/faceRecmodels/dlib_face_recognition_resnet_model_v1.dat')
imgOutfold = '/home/francesco/tmp_facce/'  

ii = 0
while(True):   
    frame = vs.read() # mutithread
    frameU = cv2.UMat(frame) # transparent api, pass to GPU mem
    grayU = cv2.cvtColor(frameU, cv2.COLOR_BGR2GRAY) # output in GPU mem
    grayU = clahe.apply(grayU)
    gray = grayU.get()
    faces,cc,dd = dlibHogdetector.run(gray,1,1)
    facesobj = dlib.full_object_detections() # array of full_object_detection objects
    for i, d in enumerate(faces): #cnn facedetect # dlib hog
        x = d.left()
        y = d.top()
        w = d.right() - x 
        h = d.bottom() - y
        dlibFaceRect = dlib.rectangle(int(x),int(y),int(x+w),int(y+h)) #haar facedetect
        shape = predictor(gray, d)
        face_descriptor = np.asarray(facerec.compute_face_descriptor(frame, shape, DATA_AUGM_N))
        facesobj.append(shape) #landmark detector)<
        images = dlib.get_face_chips(frame, facesobj, size=80, padding=0.0)   
        print(len(images))
        for kk in range(len(images)):            
            cv2.imwrite(imgOutfold + 'img'+str(ii)+"_" + str(kk) + '.png',images[kk])
            cv2.imshow("trovate",images[kk])
        ii += 1
    cv2.imshow("webcam",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


vs.stop()


print('end all')