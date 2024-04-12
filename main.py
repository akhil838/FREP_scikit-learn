from customtkinter import *
from capture_devices import devices
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import cv2
import pickle
from PIL import Image
import numpy as np
import os



camera_list = [n.replace('DEVICE NAME : ', '') for n in devices.run_with_param(device_type='video', result_=True)]


facedetect = cv2.CascadeClassifier('framework/haarcascade_frontalface_default.xml')
app = CTk()

app.bind('<Escape>', lambda e: app.quit())

default_Page = CTkFrame(app, width=300)
default_Page.grid(row=0, column=0, sticky="nsew")
default_Page_menu = CTkFrame(default_Page)
default_Page_menu.rowconfigure(1, weight=800)
default_Page_video = CTkFrame(default_Page, width=800, height=600)
default_Page_menu.grid(row=0, column=0, sticky="nsew")
default_Page_video.grid(row=0, column=1, sticky="e")

lable1 = CTkLabel(default_Page_menu, text="HOME PAGE", width=60, height=10)
lable1.grid(row=0, column=0)
space_lable = CTkLabel(default_Page_menu, text="", width=200, )
space_lable.grid(row=1, column=0)
default_Page_lable = CTkLabel(default_Page_video, text="", width=800, height=570)
# lable_video.grid(row=0, column=0, rowspan=4, sticky="nse")
image = Image.open('framework/3d-face-recognition-icon-png.webp')
photo_image = CTkImage(image, size=(400, 400))
default_Page_lable.photo_image = photo_image
default_Page_lable.configure(image=photo_image)
default_Page_lable.pack()
default_Page_lable1 = CTkLabel(default_Page_video, text="Made With ❤️ by ak838")
default_Page_lable1.pack(side=BOTTOM)
button1 = CTkButton(default_Page_menu, text="Recognise (Test)", command=lambda: (recog.tkraise()))
button1.grid(row=2, column=0, pady=5)
button6 = CTkButton(default_Page_menu, text="Add Faces (Train)", command=lambda: add_face.tkraise())
button6.grid(row=3, column=0, pady=5)

# FACE RECOGNITION FRAME
recog = CTkFrame(app)
recog_menu = CTkFrame(recog)
recog_menu.configure(width=300)
recog_menu.rowconfigure(3, weight=40)
recog_video = CTkFrame(recog, width=800, height=600)

recog.grid(row=0, column=0, sticky="nsew")
recog_menu.grid(row=0, column=0, sticky="nsew")
recog_video.grid(row=0, column=1, sticky="e")


lable2 = CTkLabel(recog_menu, text="Face Recognition", width=60, height=10)
lable2.grid(row=0, column=0, sticky="n")
lable_video = CTkLabel(recog_video, text="", width=800, height=600)
# lable_video.grid(row=0, column=0, rowspan=4, sticky="nse")
lable_video.pack()
lable0 = CTkLabel(recog_menu, text="Select Camera ", height=10)
lable0.grid(row=1, column=0,pady =5)
cam_box1 = CTkComboBox(recog_menu, state='readonly',values=camera_list)
cam_box1.grid(row=2, column=0)
space_lable = CTkLabel(recog_menu, text="", width=200)
space_lable.grid(row=3, column=0)

button2 = CTkButton(recog_menu, text="Go back to Home", command=lambda: default_Page.tkraise())
button2.grid(row=8, column=0, pady=5)

# ADD FACES FRAME
add_face = CTkFrame(app)
add_face_menu = CTkFrame(add_face)
add_face_video = CTkFrame(add_face, width=800, height=600)
add_face_menu.rowconfigure(4, weight=40)

add_face.grid(row=0, column=0, sticky="nsew")
add_face_menu.grid(row=0, column=0, sticky="nsew")
add_face_video.grid(row=0, column=1, sticky="e")

lable3 = CTkLabel(add_face_menu, text="Train a New Face", width=60, height=10)
lable3.grid(row=0, column=0, sticky="n")
lable_train = CTkLabel(add_face_video, text="", width=800, height=600)
# lable_train.grid(row=0, column=1, rowspan=4, sticky="nse")
lable_train.pack()
space_lable1 = CTkLabel(add_face_menu, text="", width=200)
space_lable1.grid(row=4, column=0)

button3 = CTkButton(add_face_menu, text="Go back to Home", command=lambda: default_Page.tkraise())
button3.grid(row=9, column=0, pady=5)
name_lable = CTkLabel(add_face_menu, text="Enter Name")
name_lable.grid(row=5, column=0)
name_input = CTkTextbox(add_face_menu, height=5, width=140)
name_input.grid(row=6, column=0)
lable0 = CTkLabel(add_face_menu, text="Select Camera ", height=10)
lable0.grid(row=2, column=0,pady =5)

cam_box2 = CTkComboBox(add_face_menu, state='readonly',values=camera_list)
cam_box2.grid(row=3, column=0)


def disablebutton(button):
    button.configure(state=DISABLED)


def enablebutton(button):
    button.configure(state=NORMAL)


def testing():
    global vid
    disablebutton(button4)
    vid = cv2.VideoCapture(camera_list.index(cam_box1.get()), cv2.CAP_DSHOW)
    width, height = 800, 600
    # Set the width and height
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

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


button4 = CTkButton(recog_menu, text="Turn On Camera", command=testing)
button4.grid(row=5, column=0, pady=5)
stop_button_recog = CTkButton(recog_menu, text="Stop", command=lambda: (vid.release(), enablebutton(button4)))
stop_button_recog.grid(row=6, column=0, pady=5)


def train():
    global vid
    disablebutton(button5)
    vid = cv2.VideoCapture(camera_list.index(cam_box2.get()), cv2.CAP_DSHOW)
    width, height = 800, 600
    # Set the width and height
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

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

            vid.release()
            enablebutton(button5)

    train_video()


button5 = CTkButton(add_face_menu, text="Train", command=train)
button5.grid(row=7, column=0, pady=5)
stop_button_recog = CTkButton(add_face_menu, text="Stop", command=lambda: (vid.release(), enablebutton(button5)))
stop_button_recog.grid(row=8, column=0, pady=5)

default_Page.tkraise()
# app.geometry("900x700")
app.title("Facial Recognition and Emotion Predictor")
app.resizable(True, False)
app.mainloop()
