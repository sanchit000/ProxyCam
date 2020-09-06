

import tkinter as tk       #import Tkinter
from tkinter import *

import cv2,os           #import opencv and os which is used for interacting with OS
import shutil           #provide fuction of copy files as well as folders
import csv
import numpy as np     #providing support of multi dimensional array and matrices
from PIL import Image, ImageTk    #provide malupulating image functions
import pandas as pd               #provide data manipulation and analysis
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font


window = tk.Tk()


window.title("ProxyCam: Smart-Attendance-System")       #title of window



window.geometry('1280x720')                        #Geometry of window
window.configure(background='#00b3b3')             #window background color







message = tk.Label(window, text="ProxyCam: Smart-Attendance-System" ,bg="black"  ,fg="white"  ,width=50  ,height=2,font=('times', 30, 'italic bold underline'))

message.place(x=50, y=20)

lbl = tk.Label(window, text="Enter ID",width=15  ,height=2  ,fg="white"  ,bg="black" ,font=('times', 15, ' bold ') )
lbl.place(x=50, y=250)

txt = tk.Entry(window,width=15  ,fg="black" ,bg="white",font=('times', 30,))
txt.place(x=350, y=250)

lbl2 = tk.Label(window, text="Enter Name",width=15  ,fg="white"  ,bg="black"    ,height=2 ,font=('times', 15, ' bold '))
lbl2.place(x=50, y=325)

txt2 = tk.Entry(window,width=15  ,fg="black"  ,bg="white",font=('times', 30, ' bold ')  )
txt2.place(x=350, y=325)

lbl3 = tk.Label(window, text="Notification : ",width=20  ,fg="red"  ,bg="yellow"  ,height=2 ,font=('times', 15, ' bold underline '))
lbl3.place(x=100, y=150)

message = tk.Label(window, text="" ,bg="yellow"  ,fg="red"  ,width=60  ,height=2, activebackground = "yellow" ,font=('times', 15, ' bold '))
message.place(x=400, y=150)

#message = tk.Label(window, text="" ,bg="black"  ,fg="white"  ,width=30  ,height=2, activebackground = "yellow" ,font=('times', 15, ' bold '))
#message.place(x=700, y=400)

#lbl3 = tk.Label(window, text="Attendance : ",width=20  ,fg="white"  ,bg="black"  ,height=2 ,font=('times', 15, ' bold  underline'))
#lbl3.place(x=400, y=650)
message2 = tk.Label(window, text="" ,fg="red"   ,bg="yellow",activeforeground = "green",width=90  ,height=2  ,font=('times', 15, ' bold '))
message2.place(x=75, y=650)



#message2 = tk.Label(window, text="" ,fg="white"   ,bg="black",activeforeground = "green",width=30  ,height=2  ,font=('times', 15, ' bold '))
#message2.place(x=700, y=650)

#photo=tk.PhotoImage(file="pass.png")
#can.create_image(0,0,image=photo,anchor=NW)

def clear():
    txt.delete(0, 'end')
    res = ""
    message.configure(text= res)

def clear2():
    txt2.delete(0, 'end')
    res = ""
    message.configure(text= res)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


#function for take image
def TakeImages():
    Id=(txt.get())
    name=(txt2.get())
    if(is_number(Id) and name.isalpha()):
        cam = cv2.VideoCapture(0)            #videon is start
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(harcascadePath)      #detect face and eyes in an image
        sampleNum=0
        while(True):
            ret, img = cam.read()                                    #capture the frame
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)            #convert BGR to gray color
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)      #create a rectangle about the face portion
                #incrementing sample number
                sampleNum=sampleNum+1
                #saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("TrainingImage\ "+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                #display the frame
                cv2.imshow('frame',img)
            #wait for 100 miliseconds
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum>60:
                break
        cam.release()
        cv2.destroyAllWindows()
        res = "Images Saved for ID : " + Id +" Name : "+ name
        row = [Id , name]
        with open('StudentDetails\StudentDetails.csv','a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text= res)
    else:
        if(is_number(Id)):
            res = "Enter Alphabetical Name"
            message.configure(text= res)
        if(name.isalpha()):
            res = "Enter Numeric Id"
            message.configure(text= res)


#function for train images.
def TrainImages():

    recognizer = cv2.face.LBPHFaceRecognizer_create()      #Face recognizer to train the dataset

    harcascadePath = "haarcascade_frontalface_default.xml"
    detector =cv2.CascadeClassifier(harcascadePath)
    faces,Id = getImagesAndLabels("TrainingImage")            #taking image one by one and returning fac and if of each image
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    res = "Image Trained"#+",".join(str(f) for f in Id)
    message.configure(text= res)

def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    #print(imagePaths)

    #create empty face list
    faces=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)
    return faces,Ids

def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()#cv2.createLBPHFaceRecognizer()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);
    df=pd.read_csv("StudentDetails\StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names =  ['Id','Name','Date','Time']
    attendance = pd.DataFrame(columns = col_names)
    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
            if(conf < 50):
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id'] == Id]['Name'].values
                tt=str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [Id,aa,date,timeStamp]

            else:
                Id='Unknown'
                tt=str(Id)
            if(conf > 75):
                noOfFile=len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])
            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)
        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')
        cv2.imshow('im',im)
        if (cv2.waitKey(1)==ord('q')):
            break
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName,index=False)
    cam.release()
    cv2.destroyAllWindows()
    #print(attendance)
    res=attendance
    message2.configure(text= res)

cl_btn=PhotoImage(file='button_clear.png')
clearButton = tk.Button(window,image=cl_btn, command=clear,bg="#addbeb", width=100, height=40,borderwidth=0)
clearButton.place(x=700, y=255)
clearButton2 = tk.Button(window,image=cl_btn, command=clear2,bg="#addbeb", width=100, height=40,borderwidth=0)
clearButton2.place(x=700, y=330)
takeImg = tk.Button(window, text="Take Images", command=TakeImages  ,fg="white"  ,bg="blue"  ,width=15  ,height=2, activebackground = "#00b3b3" ,font=('times', 15, ' bold '))
takeImg.place(x=50, y=450)
trainImg = tk.Button(window, text="Train Images", command=TrainImages  ,fg="white"  ,bg="blue"  ,width=15  ,height=2, activebackground = "#00b3b3" ,font=('times', 15, ' bold '))
trainImg.place(x=350, y=450)
trackImg = tk.Button(window, text="Take Attendance", command=TrackImages  ,fg="white"  ,bg="red"  ,width=15  ,height=2, activebackground = "#00b3b3" ,font=('times', 15, ' bold '))
trackImg.place(x=50, y=550)
quitWindow = tk.Button(window, text="Quit", command=window.destroy  ,fg="white"  ,bg="blue"  ,width=15  ,height=2, activebackground = "#00b3b3" ,font=('times', 15, ' bold '))
quitWindow.place(x=350, y=550)


window.mainloop()


