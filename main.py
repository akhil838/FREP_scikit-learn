from customtkinter import *
from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
from PIL import Image
import numpy as np
import os

vid = cv2.VideoCapture(0)
width, height = 800, 600
# Set the width and height
vid.set(cv2.CAP_PROP_FRAME_WIDTH, width)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

app = CTk()

app.bind('<Escape>', lambda e: app.quit())

default_Page = CTkFrame(app)
recog = CTkFrame(app)
recog_menu = CTkFrame(recog)
recog_menu.configure(width=300)
recog_menu.rowconfigure(1,weight=40)
recog_video = CTkFrame(recog,width=800, height=600)
add_face = CTkFrame(app)
add_face_menu = CTkFrame(add_face)
add_face_video = CTkFrame(add_face,width=800, height=600)

default_Page.grid(row=0, column=0, sticky="nsew")
recog.grid(row=0, column=0, sticky="nsew")
recog_menu.grid(row=0, column=0, sticky="nsew")
recog_video.grid(row=0, column=1, sticky="e")
add_face.grid(row=0, column=0, sticky="nsew")
add_face_menu.grid(row=0, column=0, sticky="nsew")
add_face_video.grid(row=0, column=1, sticky="e")

lable1 = CTkLabel(default_Page, text="HOME PAGE", width=60, height=10)
lable1.grid(row=0, column=0, sticky="n")
lable2 = CTkLabel(recog_menu, text="Face Recognition", width=60, height=10)
lable2.grid(row=0, column=0, sticky="n")
lable3 = CTkLabel(add_face_menu, text="Train a New Face", width=60, height=10)
lable3.grid(row=0, column=0, sticky="n")
lable_video = CTkLabel(recog_video, text="", width=800, height=600)
# lable_video.grid(row=0, column=0, rowspan=4, sticky="nse")
lable_video.pack()
lable_train = CTkLabel(add_face_video, text="", width=800, height=600)
# lable_train.grid(row=0, column=1, rowspan=4, sticky="nse")
lable_train.pack()


def testing():
    with open('data/names.pkl', 'rb') as w:
        LABELS = pickle.load(w)
    with open('data/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)
    print('Shape of Faces matrix --> ', FACES.shape)
    model = KNeighborsClassifier()
    model.fit(FACES, LABELS)

    def test_video():
        # Capture the video frame by frame
        _, frame = vid.read()
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(grey, 1.3, 5)
        # Convert image from one color space to other

        # Capture the latest frame and transform to image

        for (x, y, w, h) in faces:
            crop_img = frame[y:y + h, x:x + w, :]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
            output = model.predict(resized_img)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.putText(frame, str(output[0]), (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        # cv2.imshow("Frame",frame)
        # Convert captured image to photo image
        opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        captured_image = Image.fromarray(opencv_image)
        photo_image = CTkImage(captured_image, size=(800, 600))

        # Displaying photo image in the label
        lable_video.photo_image = photo_image

        # Configure image in the label
        lable_video.configure(image=photo_image)

        # Repeat the same process after every 10 seconds
        lable_video.after(5, test_video)

    test_video()


def train():
    global faces_data
    global i
    global name
    i = 0
    faces_data = []
    name = name_input.get("1.0", "end-1c")

    def train_video():
        global i
        global faces_data
        ret, frame = vid.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            crop_img = frame[y:y + h, x:x + w, :]
            resized_img = cv2.resize(crop_img, (50, 50))
            if len(faces_data) <= 100 and i % 10 == 0:
                faces_data.append(resized_img)
            i = i + 1
            print(f"i = {i}", len(faces_data))
            cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

        opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        captured_image = Image.fromarray(opencv_image)
        photo_image = CTkImage(captured_image, size=(800, 600))

        # Displaying photoimage in the label
        lable_train.photo_image = photo_image

        # Configure image in the label
        lable_train.configure(image=photo_image)

        # Repeat the same process after every 10 seconds
        if len(faces_data) < 100:
            lable_train.after(5, train_video)
        else:
            faces_data = np.asarray(faces_data)
            faces_data = faces_data.reshape(100, -1)

            if 'names.pkl' not in os.listdir('data/'):
                names = [name] * 100
                with open('data/names.pkl', 'wb') as f:
                    pickle.dump(names, f)
            else:
                with open('data/names.pkl', 'rb') as f:
                    names = pickle.load(f)
                names = names + [name] * 100
                with open('data/names.pkl', 'wb') as f:
                    pickle.dump(names, f)

            if 'faces_data.pkl' not in os.listdir('data/'):
                with open('data/faces_data.pkl', 'wb') as f:
                    pickle.dump(faces_data, f)
            else:
                with open('data/faces_data.pkl', 'rb') as f:
                    faces = pickle.load(f)
                faces = np.append(faces, faces_data, axis=0)
                with open('data/faces_data.pkl', 'wb') as f:
                    pickle.dump(faces, f)

    train_video()


space_lable = CTkLabel(recog_menu, text="")
space_lable.grid(row=1, column=0)

button1 = CTkButton(default_Page, text="recog", command=lambda: recog.tkraise())
button1.grid(row=2, column=0)
button6 = CTkButton(default_Page, text="add_face", command=lambda: add_face.tkraise())
button6.grid(row=3, column=0)
button2 = CTkButton(recog_menu, text="Home", command=lambda: default_Page.tkraise())
button2.grid(row=2, column=0)
button3 = CTkButton(add_face_menu, text="Home", command=lambda: default_Page.tkraise())
button3.grid(row=2, column=0)

button4 = CTkButton(recog_menu, text="camera", command=testing)
button4.grid(row=3, column=0)
button5 = CTkButton(add_face_menu, text="Train", command=train)
button5.grid(row=3, column=0)

name_input = CTkTextbox(add_face_menu, height=5, width=160)
name_input.grid(row=4, column=0)

default_Page.tkraise()
# app.geometry("900x700")
app.title("Facial Recognition and Emotion Predictor")
app.resizable(True, False)
app.mainloop()
