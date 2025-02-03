import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

import csv

import cv2
import imutils
import time
import numpy as np
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
from imutils import paths

from collections import Iterable


class MyGUI:

    def __init__(self):
        self.root = tk.Tk()

        self.root.geometry("1920x1080")
        self.root.title("Registration Window")
        self.root.configure(bg="black")

        self.name_label = tk.Label(self.root, text="Name", font=("Arial", 20))
        self.name_label.pack(padx=10, pady=10)

        self.name= tk.Entry(self.root, width=50, font=("Arial", 15))
        self.name.pack(padx=10, pady=10)

        self.rollno_label = tk.Label(self.root, text="Roll Number", font=("Arial", 20))
        self.rollno_label.pack(padx=10, pady=10)

        self.rollno = tk.Entry(self.root, width=50, font=("Arial", 15))
        self.rollno.pack(padx=10, pady=10)

        self.registerbtn = tk.Button(self.root, text="Register Profile", height=2, width=25, font=("Arial", 25), bg="white", command=self.register_csv)
        self.registerbtn.pack(padx=10, pady=10)

        self.imagebtn = tk.Button(self.root, text="Click Images", height=2, width=25, font=("Arial", 25), bg="white", command=self.click_image)
        self.imagebtn.pack(padx=10, pady=10)

        self.trainbtn = tk.Button(self.root, text="Train Model", height=2, width=25, font=("Arial", 25), bg="white", command=self.train)
        self.trainbtn.pack(padx=10, pady=10)

        self.newpagebtn = tk.Button(self.root, text="Open Attendence Interface", height=2, width=25, font=("Arial", 25), bg="white", command=self.new_window)
        self.newpagebtn.pack(padx=10, pady=10)

        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.root.mainloop()

    def register_csv(self):        
        self.Name = self.name.get()
        self.Roll_Number = self.rollno.get()
        self.info = [str(self.Name), str(self.Roll_Number)]

        with open('registration.csv', 'a') as csvFile:
            write = csv.writer(csvFile)
            write.writerow(self.info)
        csvFile.close()

        messagebox.showinfo(title="Saved", message="Registration Successful")

    def click_image(self):
        self.haar_cascade = 'model/haarcascade_frontalface_default.xml'
        self.classifier = cv2.CascadeClassifier(self.haar_cascade)

        self.dataset = 'dataset'
        self.sub_dataset = self.name.get()
        self.path = os.path.join(self.dataset, self.sub_dataset)

        if not os.path.isdir(self.path):
            os.makedirs(self.path)

        cam = cv2.VideoCapture(0)
        time.sleep(3.0)

        clicks = 0
        
        messagebox.showinfo(title="Camera is opening", message="Get Ready")
        
        while clicks < 51:
            print(clicks)
            _,img = cam.read()
            img = imutils.resize(img, width=400)
            self.face = self.classifier.detectMultiScale(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), scaleFactor=1.1,
                minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in self.face:
                #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                clicked_img = os.path.sep.join([self.path, "{}.png".format(
                    str(clicks).zfill(5))])
                cv2.imwrite(clicked_img, img)
                cv2.putText(img, str(clicks), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0))
                clicks += 1
            
            
            cv2.imshow("Frame", img)

            self.key = cv2.waitKey(1) & 0xFF
            if self.key == ord("q"):
                break

        cam.release()
        cv2.destroyAllWindows()

        print("Dataset Creation Successful")
        messagebox.showinfo(title="Camera Closed", message="Dataset Creation Successful")


    def train(self):
        messagebox.showinfo(title="Training Started", message="Training....")
        
        self.dataset = "dataset"

        embedding_file = "output/embeddings.pickle"
        embedding_model = "model/openface_nn4.small2.v1.t7"
        prototxt = "model/deploy.prototxt"
        model =  "model/res10_300x300_ssd_iter_140000.caffemodel"

        dnndetector = cv2.dnn.readNetFromCaffe(prototxt, model)
        dnnembedder = cv2.dnn.readNetFromTorch(embedding_model)

        image_path = list(paths.list_images(self.dataset))

        knownEmbeddings = []
        knownNames = []

        total = 0
        conf = 0.5

        for (i, imagePath) in enumerate(image_path):
            print("Processing image {}/{}".format(i + 1,len(image_path)))
            student_name = imagePath.split(os.path.sep)[-2]
            image = cv2.imread(imagePath)
            image = imutils.resize(image, width=600)
            (h, w) = image.shape[:2]
            
            blob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)

            dnndetector.setInput(blob)
            detections = dnndetector.forward()

            if len(detections) > 0:
                i = np.argmax(detections[0, 0, :, 2])
                confidence = detections[0, 0, i, 2]

                if confidence > conf:
                    rect = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = rect.astype("int")
                    faceOnly = image[startY:endY, startX:endX]
                    (fH, fW) = faceOnly.shape[:2]

                    if fW < 20 or fH < 20:
                        continue
                    
                    faceBlob = cv2.dnn.blobFromImage(faceOnly, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                    dnnembedder.setInput(faceBlob)
                    
                    vec = dnnembedder.forward()
                    knownNames.append(student_name)
                    knownEmbeddings.append(vec.flatten())
                    total += 1

        print("Embedding:{0} ".format(total))
        data = {"embeddings": knownEmbeddings, "names": knownNames}
        f = open(embedding_file, "wb")
        f.write(pickle.dumps(data))
        f.close()

        embedding_file = "output/embeddings.pickle"
        recognizer_file = "output/recognizer.pickle"
        labelEncoder_file = "output/le.pickle"

        pickledata = pickle.loads(open(embedding_file, "rb").read())

        labelEnc = LabelEncoder()
        labels = labelEnc.fit_transform(pickledata["names"])

        recognizer = SVC(C=1.0, kernel="linear", probability=True)
        recognizer.fit(pickledata["embeddings"], labels)

        f = open(recognizer_file, "wb")
        f.write(pickle.dumps(recognizer))
        f.close()

        f = open(labelEncoder_file, "wb")
        f.write(pickle.dumps(labelEnc))
        f.close()

        print("Process Completed")

        messagebox.showinfo(title="Training Ended", message="Training Succesful")

    def new_window(self):
        self.newpage = tk.Toplevel()
        self.newpage.geometry("1920x1080")
        self.newpage.configure(bg="black")
        self.newpage.title("Attendence Interface")

        self.recognizebtn = tk.Button(self.newpage, text="Recognize", height=2, width=25, font=("Arial", 25), bg = "white",command=self.recognize)
        self.recognizebtn.place(x=100, y = 25)
        #self.recognizebtn.pack(padx=10, pady=10)

        self.recordbtn = tk.Button(self.newpage, text="Record Attendence", height=2, width=25, font=("Arial", 25), bg="white", command=self.record)
        self.recordbtn.place(x=900, y=25)
        #self.recordbtn.pack(padx=10, pady=10)

        self.closebtn = tk.Button(self.newpage, text="Close Streaming", height=2, width=25, font=("Arial", 25), bg="white", command=self.close_video)
        self.closebtn.place(x=900, y=600)

        self.frame = tk.LabelFrame(self.newpage, height=600, width=600, text="Frame", bg="white")
        self.frame.place(x=50, y = 150)
        #self.frame.pack(padx=10, pady=10)
        self.l = tk.Label(self.frame, height=600, width=600)
        self.l.place(x=0, y = 0)

        self.finalname = tk.Label(self.newpage, text=str(self.opname), font=("Arial", 25))
        self.finalname.place(x=975, y=250)
        #self.finalname.pack(padx=10, pady=10)

        self.finalrollno = tk.Label(self.newpage, text=str(self.opRoll_Number), font=("Arial", 25))
        self.finalrollno.place(x=975, y=350)
        #self.finalrollno.pack(padx=10, pady=10)

        #self.clearbtn = tk.Button(self.newpage, text="Clear", height=2, width=25, font=("Arial", 25), bg="white", command=self.clear)
        #self.clearbtn.pack(padx=10, pady=10)

    def recognize(self):
        
        messagebox.showinfo(title="Starting Recognition", message="Get Ready...")
        def flatten(itemlist):
            for item in itemlist:
                if isinstance(item, Iterable) and not isinstance(item, str):
                    for x in flatten(item):
                        yield x
                else:
                    yield item


        embedding_file = "output/embeddings.pickle"
        embedding_model = "model/openface_nn4.small2.v1.t7"
        recognizer_file = "output/recognizer.pickle"
        labelEncoder_file = "output/le.pickle"
        conf = 0.5
        prototxt = "model/deploy.prototxt"
        model = "model/res10_300x300_ssd_iter_140000.caffemodel"

        dnndetector = cv2.dnn.readNetFromCaffe(prototxt, model)
        dnnembedder = cv2.dnn.readNetFromTorch(embedding_model)

        recognizer = pickle.loads(open(recognizer_file, "rb").read())
        le = pickle.loads(open(labelEncoder_file, "rb").read())

        self.opRoll_Number = ""
        box = []

        cam = cv2.VideoCapture(0)
        time.sleep(3.0)

        while True:
            _, img = cam.read()
            img = imutils.resize(img, width=600)
            (h, w) = img.shape[:2]
            blob = cv2.dnn.blobFromImage(
                cv2.resize(img, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0), swapRB=False, crop=False)

            dnndetector.setInput(blob)
            detections = dnndetector.forward()

            for i in range(0, detections.shape[2]):

                confidence = detections[0, 0, i, 2]

                if confidence > conf:

                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    faceOnly = img[startY:endY, startX:endX]
                    (fH, fW) = faceOnly.shape[:2]

                    if fW < 20 or fH < 20:
                        continue

                    faceBlob = cv2.dnn.blobFromImage(faceOnly, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                    dnnembedder.setInput(faceBlob)
                    vec = dnnembedder.forward()

                    preds = recognizer.predict_proba(vec)[0]
                    j = np.argmax(preds)
                    self.proba = preds[j]
                    self.opname = le.classes_[j]

                    with open('registration.csv', 'r') as csvFile:
                        reader = csv.reader(csvFile)
                        for row in reader:
                            box = np.append(box, row)
                            self.opname = str(self.opname)
                            if self.opname in row:
                                person = str(row)
                                print(self.opname)
                        listString = str(box)
                        #print(box)
                        if self.opname in listString:
                            singleList = list(flatten(box))
                            listlength = len(singleList)
                            index = singleList.index(self.opname)
                            self.opname = singleList[index]
                            self.opRoll_Number = singleList[index + 1]
                            print(self.opRoll_Number)

                    text = "{} : {} : {:.2f}%".format(self.opname, self.opRoll_Number, self.proba * 100)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    
                    cv2.rectangle(img, (startX, startY), (endX, endY),
                                (0, 0, 255), 2)
                    cv2.putText(img, text, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            
            #cv2.imshow("Frame", img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(img))
            self.l["image"] = img

            self.newpage.update()
            
            # key = cv2.waitKey(1) & 0xFF
            # if key == 27:
            #     break

        # cam.release()
        # cv2.destroyAllWindows()
        
        #messagebox.showinfo(title="Recognition Succesful", message="Please Record Your Attendence.")
    def close_video(self):
        self.newpage.destroy()
        messagebox.showinfo(title="Attendence Recorded", message="Session has been completed")

    def record(self):
        self.recname = self.opname
        self.recRoll_Number = self.opRoll_Number
        self.recinfo = [str(self.recname), str(self.recRoll_Number)]

        with open('attendence.csv', 'a') as csvFile:
            write = csv.writer(csvFile)
            write.writerow(self.recinfo)
        csvFile.close()

        messagebox.showinfo(title="Attendence Status", message="Attendence has been recorded Successfully.")

    #def clear(self):
        #self.recognizedinfo.delete('1.0', tk.END)

    def close(self):
        if messagebox.askyesno(title="Quit?", message="Do you want to quit ?"):
            self.root.destroy()

MyGUI()