import os 
import cv2
import face_recognition
from PIL import Image
import numpy as np

np_load_old = np.load
np.load.__defaults__=(None, True, True, 'ASCII')


modelFile ="opencvdnnfp16/res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "opencvdnnfp16/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)



#function to extract box dimensions
def face_dnn(img, coord=False):
    blob = cv2.dnn.blobFromImage(img, 1.2, (224,224), [104, 117, 123], False, False) #
    # params: source, scale=1, size=300,300, mean RGB values (r,g,b), rgb swapping=false, crop = false
    conf_threshold=0.6 # confidence at least 60%
    frameWidth=img.shape[1] # get image width
    frameHeight=img.shape[0] # get image height
    max_confidence=0
    net.setInput(blob)
    detections = net.forward()
    detection_index=0
    bboxes = []
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            
            if max_confidence < confidence: # only show maximum confidence face
                max_confidence = confidence
                detection_index = i
    i=detection_index        
    x1 = int(detections[0, 0, i, 3] * frameWidth)
    y1 = int(detections[0, 0, i, 4] * frameHeight)
    x2 = int(detections[0, 0, i, 5] * frameWidth)
    y2 = int(detections[0, 0, i, 6] * frameHeight)
    #cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,0),2)
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if coord==True:
        return x1, y1, x2, y2
    return cv_rgb



def extract_face(filename, required_size=(160, 160)):
    image = Image.open(filename)
    #image = image.convert('RGB')
    pixels = np.asarray(image)
    x1, y1, width, height = face_dnn(pixels, coord=True)
    #print(x1, y1, width, height)
    x1, y1 = abs(x1), abs(y1)
    #x2, y2 = x1 + width, y1 + height
    face = pixels[y1:height, x1:width]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array  



def getImagesAndLabels(path):
    names = os.listdir(path)
    current_id = 0
    label_ids = {}
    #ids = []
    
    imagePaths = []
    for n in names:
        if "." in n:
            continue
        for f in os.listdir(path + n):
            imPath = os.path.join(path + n, f)
            imagePaths.append(imPath)    

    faceSamples=[]
    
    for imagePath in imagePaths:
        
        cur_img = cv2.imread(imagePath)
        faceSamples.append(cur_img)
        nm = str(os.path.split(imagePath)[0].split("/")[1])
        
        
        #PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        
        #img_numpy = np.array(PIL_img,'uint8')
        
        if not current_id in label_ids:
            label_ids[current_id] = nm
            current_id += 1
        
        #id_ = label_ids[nm]

        #faces = detector.detectMultiScale(img_numpy)
        #for (x,y,w,h) in faces:
            #faceSamples.append(img_numpy[y:y+h,x:x+w])
            #ids.append(id_)
    return faceSamples, label_ids



def findEncodings(images):
    encodeList = []
    for img in images:
        if img is None:
            continue
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        except IndexError:
            continue
        
    return encodeList


def pred_img(encodeFace, encoded_faces_path='face_encodings.npz'):
    
    #np.load.__defaults__=(None, True, True, 'ASCII')
    old = np.load
    np.load = lambda *a,**k: old(*a,allow_pickle=True)
    
    encoded_faces = np.load(encoded_faces_path, allow_pickle=True)
    
    #np.load.__defaults__=(None, False, True, 'ASCII')
    
    encodeListKnown = encoded_faces['arr_0']
    labels = encoded_faces['arr_1'].item()
    #labels = 
    
    
    matches = face_recognition.compare_faces(encodeListKnown, encodeFace, tolerance=0.5)
    faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
    
    matchIndex = np.argmin(faceDis)
    
    if matches[matchIndex]:
        name = labels[matchIndex].upper()
    else:
        name = 'Unknown'
        
    return name


def update_face_encodings(file_name='face_encodings.npz'):

    path = 'images/'

    faces, labels = getImagesAndLabels(path)

    encodeListKnown = findEncodings(faces)
    #print('Encoding Complete')
    #print(len(encodeListKnown))

    #print(encodeListKnown[0])

    np.savez_compressed(file_name, encodeListKnown, labels)
    
    return True
    
    



np.load.__defaults__=(None, False, True, 'ASCII')

