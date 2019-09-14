import cv2
from keras.models import load_model
import os
import numpy as np
import pyttsx3

model = load_model('Final_model_asl.h5')

model_d = load_model('digit_model.h5')

dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',
                   13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',
                   25:'Z',26:'space',27:'del',28:'nothing'}

dict_digit = {29:'0',38:'1',39:'2',45:'3',46:'4',42:'5',44:'6',41:'7',43:'8',40:'9',30:'Best of luck',
              31:'Fuck',32:'I Love You',33:'Like',34:'Love',35:'Me',36:'Remember',37:'You'}

def text_to_voice(message):
    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
    engine.setProperty('rate',rate-50)
    voices = engine.getProperty('voices')
    engine.setProperty('voice',voices[1].id)
    engine.say(message)
    engine.runAndWait()

def predict_digit(image):
    image = cv2.resize(image , (64,64))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    _, image = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY)
    image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    image = image.astype('float32')/255.0
    image = image.reshape(1,64,64,3)
    pred = model_d.predict_classes(image)
    return pred

def predict(image):
    image = cv2.resize(image , (64,64))
    image = image.astype('float32')/255.0
    image = image.reshape(1,64,64,3)
    pred = model.predict_classes(image)
    return pred

s = ""
cap = cv2.VideoCapture(0)

while True:


    ret , frame = cap.read()
    frame = cv2.resize(frame,(1200,700))
    frame = cv2.flip(frame,1)
    x1, y1, x2, y2 = 700, 100, 1100, 500
    img_cropped = frame[y1:y2, x1:x2]
    cv2.line(frame,(x1,y1),(x1,y2),(255,0,0),2)
    cv2.line(frame,(x2,y1),(x2,y2),(255,0,0),2)
    cv2.line(frame,(x1,y1),(x2,y1),(255,0,0),2)
    cv2.line(frame,(x1,y2),(x2,y2),(255,0,0),2)
    cv2.imshow('FRAME',frame)

    k = cv2.waitKey(1)
    # if cv2.waitKey(1) & 0xFF == ord("c"):

    if k == ord('c'):
        preds = predict(img_cropped)
        # print(dict[preds[0]])

        p = dict[preds[0]]
        text_to_voice(p)
        if p == 'space':
            s = s+' '
        elif p == 'del':
            s = s[:-1]
        else:
            s=s+p
        print('String  ',s)

    if k==ord('d'):
        preds_d = predict_digit(img_cropped)
        print(dict_digit[preds_d[0]])

        p = dict_digit[preds_d[0]]

        text_to_voice(p)

        s = s+p
        print("String2",s)

    # try:
    #     cv2.putText(frame,str(preds),(500,300),cv2.FONT_HERSHEY_SIMPLEX,4,(255,0,0),4)
    # except:
    #     pass

    if k==ord('q'):
        break

    cv2.putText(frame,s,(500,300),cv2.FONT_HERSHEY_SIMPLEX,4,(255,0,0),4)
cap.release()
cv2.destroyAllWindows()

# try:
print("YOYOYOYOYOYOYOY")
print(s)
text_to_voice(s)
# except:
#     pass

# try:
#     print(preds)
# except:
#     pass

test_dir = 'dataset/asl_alphabet_test/asl_alphabet_test'

def load_test_data():
    images = []
    names = []
    size = 64,64
    for image in os.listdir(test_dir):
        temp = cv2.imread(test_dir + '/' + image)
        temp = cv2.resize(temp, size)
        images.append(temp)
        names.append(image)
    images = np.array(images)
    images = images.astype('float32')/255.0
    return images, names

test_images, test_img_names = load_test_data()

def give_predictions(test_data):
    predictions_classes = []
    for image in test_data:
        image = image.reshape(1,64,64,3)
        pred = model.predict_classes(image)
        predictions_classes.append(pred[0])
    return predictions_classes

# predictions = give_predictions(test_images)
# print(predictions)


# dir = 'blackmamba/ASL/test/'
# ll =[]

# for i in os.listdir('blackmamba/ASL/test/'):
#     image = cv2.imread(dir+i)
#     image = cv2.resize(image , (64,64))
#     image = image.astype('float32')/255.0
#     image = image.reshape(1,64,64,3)
#     pred = model_d.predict_classes(image)
#     ll.append(pred)
#
# print(ll)
# print(len(ll))