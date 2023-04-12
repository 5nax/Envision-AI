############################################# IMPORTING ################################################
import datetime
import json
import textwrap
import threading
import time
import tkinter as tk
import webbrowser
from tkinter import *
from tkinter import messagebox as mess
from tkinter.messagebox import askyesno
import tkinter.ttk as ttk
import cv2
import pyttsx3
import requests
import speech_recognition as sr
import wikipedia
from PIL import Image, ImageFilter
from PIL import ImageTk
import os
from tkinter import messagebox
from yolov5 import YOLOv5
import torch
from torchvision.ops import nms
import queue
import face_recognition
import pickle
from mtcnn import MTCNN
from tkinter import filedialog
import shutil
import numpy as np
import glob
import subprocess
import platform

###########################################################################################

language = 'en'

engine = pyttsx3.init('sapi5')

voices = engine.getProperty('voices')

engine.setProperty('voice', voices[1].id)

rate = engine.getProperty('rate')

engine.setProperty('rate', 130)

assistant_running = False

running = True

camera_initialized = False

ts = time.time()

date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')

day, month, year = date.split("-")

mont = {'01': 'January',
        '02': 'February',
        '03': 'March',
        '04': 'April',
        '05': 'May',
        '06': 'June',
        '07': 'July',
        '08': 'August',
        '09': 'September',
        '10': 'October',
        '11': 'November',
        '12': 'December'
        }


def tick():
    time_string = time.strftime('%I:%M:%S %p')
    clock.config(text=time_string)
    clock.after(200, tick)


###########################################################################################

def speak(audio):
    engine.say(audio)
    engine.runAndWait()


def speak1(response):
    engine.say(response)
    engine.runAndWait()


def speak_label(label, direction, engine,stop_event):
    if not stop_event.is_set():
        engine.say(f"{label} on your {direction}")
        engine.runAndWait()


def take_command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        user_label.config(text="Listening...")
        user_label.update()
        print("Listening...")
        r.pause_threshold = 1
        audio = r.listen(source)
    try:
        print("Recognizing...")
        user_label.config(text="Recognizing...")
        user_label.update()
        query = r.recognize_google(audio, language='en-us')
        print(f"User said: {query}\n")
    except Exception as e:
        print(e)
        return "Speech Recognition could not understand audio"
    return query


####################################### Computer Vision Person Register ####################################################

def add_new_face(name):
    if name:
        if capture_image_and_save(name):
            new_face_encoding, _ = encode_faces([f"{name}.jpg"], [name])
            known_face_encodings, known_face_names = load_known_faces("known_faces.pkl")
            known_face_encodings.extend(new_face_encoding)
            known_face_names.extend([name])

            save_known_faces(known_face_encodings, known_face_names, "known_faces.pkl")
            messagebox.showinfo("Success", f"{name} has been added successfully!")
            thumbnail_treeview.update_list()

def capture_image_and_save(name):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        cv2.imshow("Press 's' to save, 'q' to quit", frame)
        key = cv2.waitKey(1)

        if key == ord('s'):
            temp_image_path = 'temp_image.jpg'
            cv2.imwrite(temp_image_path, frame)
            temp_image = face_recognition.load_image_file(temp_image_path)
            face_encodings = face_recognition.face_encodings(temp_image)

            if len(face_encodings) > 0:
                cv2.imwrite(f"{name}.jpg", frame)
                os.remove(temp_image_path)
                break
            else:
                messagebox.showerror("Error", "No face detected in the captured image. Please try again.")
                os.remove(temp_image_path)
                continue
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return False

    cap.release()
    cv2.destroyAllWindows()
    return True


def add_new_face_from_image(name):
    if name:
        image_path = filedialog.askopenfilename(title="Select the image of the person",
                                                filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

        if not image_path:
            messagebox.showerror("Error", "No image selected.")
            return

        temp_image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(temp_image)

        if len(face_encodings) > 0:
            shutil.copy(image_path, f"{name}.jpg")
            new_face_encoding, _ = encode_faces([f"{name}.jpg"], [name])
            known_face_encodings, known_face_names = load_known_faces("known_faces.pkl")
            known_face_encodings.extend(new_face_encoding)
            known_face_names.extend([name])

            save_known_faces(known_face_encodings, known_face_names, "known_faces.pkl")
            messagebox.showinfo("Success", f"{name} has been added successfully!")
            thumbnail_treeview.update_list()
        else:
            messagebox.showerror("Error", "No face detected in the selected image. Please try another image.")

def encode_faces(image_paths, names):
    known_face_encodings = []
    known_face_names = []

    for image_path, name in zip(image_paths, names):
        image = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(name)

    return known_face_encodings, known_face_names


def save_known_faces(known_face_encodings, known_face_names, filename):
    data = {
        "encodings": known_face_encodings,
        "names": known_face_names
    }
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def load_known_faces(filename):
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        return [], []

    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data["encodings"], data["names"]

def load_known_faces1():
    global known_faces
    if os.path.exists("known_faces.pkl"):
        with open("known_faces.pkl", "rb") as f:
            known_faces = pickle.load(f)
    else:
        known_faces = {}

load_known_faces1()

known_face_encodings, known_face_names = load_known_faces("known_faces.pkl")


def update_known_face(name, filename):
    known_face_encodings, known_face_names = load_known_faces(filename)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Press 's' to save the current frame or 'q' to quit", frame)
        key = cv2.waitKey(1)

        if key == ord('s'):
            cv2.imwrite('temp_image.jpg', frame)
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    temp_image_path = 'temp_image.jpg'
    temp_image = face_recognition.load_image_file(temp_image_path)
    face_encodings = face_recognition.face_encodings(temp_image)

    if len(face_encodings) > 0:
        new_face_encoding = face_encodings[0]

        face_index = known_face_names.index(name)
        known_face_encodings[face_index] = new_face_encoding

        save_known_faces(known_face_encodings, known_face_names, filename)
        os.remove(temp_image_path)
        messagebox.showinfo("Success", f"{name} has been updated successfully!")
    else:
        messagebox.showerror("Error", "No face detected in the captured image. Please try again.")
        os.remove(temp_image_path)

def update_known_face_button(name):
    global known_face_encodings, known_face_names
    if name is not None:
        if name in known_face_names:
            update_known_face(name, "known_faces.pkl")
            known_face_encodings, known_face_names = load_known_faces("known_faces.pkl")
            messagebox.showinfo("Update Known Face", f"Updated the image for {name}.")
        else:
            messagebox.showerror("Update Known Face", f"No saved image found for {name}.")


def load_known_faces2(pkl_file):
    with open(pkl_file, "rb") as f:
        known_faces = pickle.load(f)
    known_face_encodings, known_face_names = zip(*known_faces)
    return known_face_encodings, known_face_names


def detect_known_faces(face_detector, known_face_encodings, known_face_names, frame):
    face_locations = []
    face_names = []

    # Convert the image from BGR to RGB (as face_recognition uses the RGB format)
    rgb_frame = frame[:, :, ::-1]


    # Detect faces in the frame
    face_detections = face_detector.detect_faces(rgb_frame)

    for face in face_detections:
        x, y, w, h = face['box']
        top, right, bottom, left = y, x + w, y + h, x
        face_locations.append((top, right, bottom, left))

        # Extract the face encoding
        face_encoding = face_recognition.face_encodings(rgb_frame, [(top, right, bottom, left)])[0]

        # Match the encoding with the known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        face_names.append(name)

    return face_locations, face_names


def delete_face(name):
    known_face_encodings, known_face_names = load_known_faces("known_faces.pkl")
    try:
        index = known_face_names.index(name)
        del known_face_encodings[index]
        del known_face_names[index]
        save_known_faces(known_face_encodings, known_face_names, "known_faces.pkl")
        os.remove(f"{name}.jpg")
        messagebox.showinfo("Success", f"{name} has been deleted successfully!")
    except ValueError:
        messagebox.showerror("Error", f"{name} not found in the known faces!")


def delete_all_known_faces(filename):
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    data = {"encodings": known_face_encodings, "names": known_face_names}
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    messagebox.showinfo("Delete All Known Faces", "All known faces have been deleted.")


def get_direction(x, y, w, h, width, height):
    """
    Calculates the direction of the object based on its position in the frame
    """
    x_center = x + w / 2
    y_center = y + h / 2

    if x_center < width / 3:
        direction = "left"
    elif x_center > 2 * width / 3:
        direction = "right"
    else:
        direction = "center"

    if y_center < height / 3:
        direction += " top"
    elif y_center > 2 * height / 3:
        direction += " bottom"
    else:
        direction += " middle"

    return direction

######################################## Computer Vision Detection ###################################################
def detect_objects(frame, model):
    results = model.predict(frame)
    boxes = results.xyxy[0][:, :4]  # Extract only the x1, y1, x2, y2 values
    class_ids = results.xyxy[0][:, 5].long()  # Convert class IDs to integer
    confidences = results.xyxy[0][:, 4]

    # Apply confidence threshold
    min_confidence = 0.5
    mask = confidences > min_confidence
    boxes = boxes[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    # Apply Non-Maximum Suppression
    device = torch.device("cuda:0")
    box_tensors = torch.tensor(boxes).to(device)
    confidence_tensors = torch.tensor(confidences).to(device)
    nms_indices = nms(box_tensors, confidence_tensors, iou_threshold=0.4)

    indexes = nms_indices.tolist()
    return indexes, boxes, confidences, class_ids

def face_detection_thread(face_detector, known_face_encodings, known_face_names, frame_queue, face_info_queue, exit_event):
    while not exit_event.is_set():
        frame = frame_queue.get()
        face_locations, face_names = detect_known_faces(face_detector, known_face_encodings, known_face_names, frame)
        face_info_queue.put((face_locations, face_names))


def apply_night_vision(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_frame = clahe.apply(gray_frame)

    return cv2.cvtColor(equalized_frame, cv2.COLOR_GRAY2BGR)

def is_frame_too_dark(frame, threshold=30):
    mean_brightness = np.mean(frame)
    return mean_brightness < threshold

def run_computer_vision():
    global face_locations, face_names
    lmain = tk.Label(CVFrame1,
                     text="Computer Vision Started \n Press the key 'Q' on your keyboard to stop\n  Computer Vision",
                     font=('Century Gothic', 14), bg="#D0D3D4", height=30, width=50)
    lmain.place(relx=0, rely=0)
    face_detector = MTCNN()

    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    device = torch.device("cuda:0")
    model = YOLOv5('\\yolov5x.pt', device=device)

    cap = cv2.VideoCapture(0)
    cooldown_time = 4
    spoken_labels = {}

    known_face_encodings, known_face_names = load_known_faces("known_faces.pkl")
    skip_frames = 0
    max_skip_frames = 10
    stop_event = threading.Event()

    frame_queue = queue.Queue()
    face_info_queue = queue.Queue()
    face_detection_exit_event = threading.Event()
    face_detection_thread_obj = threading.Thread(target=face_detection_thread,
                                                 args=(
                                                     face_detector, known_face_encodings, known_face_names, frame_queue,
                                                     face_info_queue, face_detection_exit_event),
                                                 daemon=True)
    face_detection_thread_obj.start()

    last_confidence_update_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        scale = 0.5
        frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))

        if is_frame_too_dark(frame):
            frame = apply_night_vision(frame)

        indexes, boxes, confidences, class_ids = detect_objects(frame, model)
        frame_queue.put(frame.copy())

        if not face_info_queue.empty():
            face_locations, face_names = face_info_queue.get()

        if skip_frames == 0:
            face_locations, face_names = detect_known_faces(face_detector, known_face_encodings, known_face_names,
                                                            frame)
        skip_frames += 1
        if skip_frames > max_skip_frames:
            skip_frames = 0

        for idx in indexes:
            x, y, w, h = boxes[idx]
            x, y, w, h = int(x), int(y), int(w), int(h)
            label = classes[int(class_ids[idx])]

            current_time = time.time()
            if current_time - last_confidence_update_time >= 1:
                confidence = confidences[idx] * 100
                last_confidence_update_time = current_time

            if label == "person":
                found_recognized_face = False
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    if x <= left <= x + w and y <= top <= y + h:
                        if name != "Unknown":
                            label = name
                            found_recognized_face = True
                        break

            direction = get_direction(x, y, w, h, frame.shape[1], frame.shape[0])

            if label == "person" and found_recognized_face:
                label_with_direction_confidence = f"{label} ({direction}) {confidence:.2f}%"
            else:
                label_with_direction_confidence = f"{label} ({direction}) {confidence:.2f}%"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame,label_with_direction_confidence, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)

        if (label not in spoken_labels) or (current_time - spoken_labels[label]) > cooldown_time:
            threading.Thread(target=speak_label, args=(label, direction, engine, stop_event), daemon=True).start()
            spoken_labels[label] = current_time

        display_scale = 1.2
        display_frame = cv2.resize(frame, (int(frame.shape[1] * display_scale), int(frame.shape[0] * display_scale)))

        cv2.imshow("Detection", display_frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('image.jpg', frame)

    face_detection_exit_event.set()

    cap.release()
    cv2.destroyAllWindows()
    Camera()


def computervision1():
    global running
    running = True
    computer_vision_thread = threading.Thread(target=run_computer_vision, daemon=True)
    computer_vision_thread.start()
    speak("Starting Computer Vision")


def grab_frames(cap, frame_queue):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put(frame)



def Camera():
    global camera_initialized
    if camera_initialized:
        return
    head3 = tk.Label(CVFrame1, text="                        Camera Feed Section                            ",
                     fg="White",
                     bg="#424949", font=('Century Gothic', 17), height=1)
    head3.place(x=0, y=0)

    # Capture from camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        # Create a label in the frame to display "No camera connected"
        lmain = tk.Label(CVFrame1, text="No camera connected.. \n Connect a camera to start computer vision",
                         font=('Century Gothic', 14), bg="#D0D3D4")
        lmain.place(relx=0.5, rely=0.5, anchor="center")
    else:
        # Create a label in the frame for displaying the camera feed
        lmain = tk.Label(CVFrame1, highlightthickness=0, bd=0)
        lmain.place(x=-10, y=55)

        # Frame queue
        frame_queue = queue.Queue(maxsize=5)

        # start the frame grabbing thread
        grab_thread = threading.Thread(target=grab_frames, args=(cap, frame_queue))
        grab_thread.daemon = True
        grab_thread.start()

        # function for updating the label with the frame
        def update_label():
            try:
                frame = frame_queue.get(timeout=0.1)
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                img = Image.fromarray(cv2image)
                tk_image = ImageTk.PhotoImage(image=img)
                lmain.configure(image=tk_image)
                lmain.image = tk_image  # prevent garbage collection
            except queue.Empty:
                pass

            # schedule the update to happen again after 10ms
            lmain.after(10, update_label)

        # start the label update loop in the main thread
        update_label()
        camera_initialized = True


########################################################################################
def process_input():
    while assistant_running:
        query = take_command().lower()
        if query == "Speech Recognition could not understand audio":
            ai_response = "Sorry, I could not understand your voice. Please try again."
            response_label.config(text=textwrap.fill(ai_response, width=100))
            AssistantFrame.update()
            speak(ai_response)
            continue  # skip the rest of the loop and start listening again
        user_label.config(text=query.capitalize())
        AssistantFrame.update()
        if 'hello' in query:
            ai_response = "Hello! Please tell me how can I help you."
            response_label.config(text=textwrap.fill(ai_response, width=100))
            AssistantFrame.update()
            speak(ai_response)
            while True:
                query = take_command().lower()
                if query == "Speech Recognition could not understand audio":
                    ai_response = "Sorry, I could not understand your voice. Please try again."
                    response_label.config(text=textwrap.fill(ai_response, width=100))
                    AssistantFrame.update()
                    speak(ai_response)
                    user_label.config(text=query.capitalize())
                    AssistantFrame.update()
                    continue
                elif 'search' in query:
                    ai_response = "What do you want me to search for?"
                    response_label.config(text=textwrap.fill(ai_response, width=100))
                    AssistantFrame.update()
                    speak(ai_response)
                    query = take_command().lower()
                    if query == "speech recognition could not understand audio":
                        ai_response = "Sorry, I could not understand your voice. Please try again."
                        response_label.config(text=textwrap.fill(ai_response, width=100))
                        AssistantFrame.update()
                        speak(ai_response)
                        continue
                    user_label.config(text=query.capitalize())
                    AssistantFrame.update()
                    try:
                        ai_response = "Searching for " + query + "..."
                        response_label.config(text=textwrap.fill(ai_response, width=100))
                        AssistantFrame.update()
                        speak(ai_response)
                        query = query.replace("wikipedia", '')
                        results = wikipedia.summary(query, sentences=2)
                        response_label.config(text=textwrap.fill(results, width=100))
                        AssistantFrame.update()
                        speak(results)
                    except wikipedia.exceptions.PageError:
                        ai_response = f"Sorry, I couldn't find any information about {query} on Wikipedia."
                        response_label.config(text=textwrap.fill(ai_response, width=100))
                        AssistantFrame.update()
                        speak(ai_response)
                    except wikipedia.exceptions.DisambiguationError as e:
                        options = e.options[:5]  # limit options to the top 5
                        ai_response = "There are multiple options. Here are the top 5 results:\n\n"
                        for i, option in enumerate(options):
                            ai_response += f"{i + 1}. {option}\n"
                        ai_response += "\n. Please provide a more specific search term or say 'exit' to cancel."
                        response_label.config(text=textwrap.fill(ai_response, width=100))
                        AssistantFrame.update()
                        speak(ai_response)
                        while True:
                            query = take_command().lower()
                            user_label.config(text=query.capitalize())
                            AssistantFrame.update()
                            if query == 'exit':
                                ai_response = "Search cancelled."
                                response_label.config(text=textwrap.fill(ai_response, width=100))
                                AssistantFrame.update()
                                speak(ai_response)
                                break
                            try:
                                ai_response = "Searching for " + query + "..."
                                response_label.config(text=textwrap.fill(ai_response, width=100))
                                AssistantFrame.update()
                                speak(ai_response)
                                query = query.replace("wikipedia", '')
                                results = wikipedia.summary(query, sentences=2)
                                response_label.config(text=textwrap.fill(results, width=100))
                                AssistantFrame.update()
                                speak(results)
                                break  # exit loop if results are found
                            except wikipedia.exceptions.DisambiguationError as e:
                                options = e.options[:5]  # limit options to the top 5
                                ai_response = "There are still multiple options. Please provide a more specific search term or type 'exit' to cancel."
                                response_label.config(text=textwrap.fill(ai_response, width=100))
                                AssistantFrame.update()
                                speak(ai_response)
                                response_label.config(text=textwrap.fill(options, width=100))
                                AssistantFrame.update()
                                speak(options)

                                for option in options:
                                    response_label.config(text=textwrap.fill(option, width=100))
                                    AssistantFrame.update()
                                    speak(option)
                                    response_label.config(text=textwrap.fill(" ", width=100))
                            except wikipedia.exceptions.PageError:
                                ai_response = "Sorry, I could not find any results for your search term. Please provide a different search term or type 'exit' to cancel."
                                response_label.config(text=textwrap.fill(ai_response, width=100))
                                AssistantFrame.update()
                                speak(ai_response)

                elif 'who are you' in query:
                    ai_response = "I am Sense.\nDeveloped by Himal."
                    response_label.config(text=textwrap.fill(ai_response, width=100))
                    AssistantFrame.update()
                    speak(ai_response)

                elif 'nice to meet you' in query:
                    ai_response = "Nice to meet you too!"
                    response_label.config(text=textwrap.fill(ai_response, width=100))
                    AssistantFrame.update()
                    speak(ai_response)

                elif 'open youtube' in query:
                    ai_response = "Opening YouTube."
                    response_label.config(text=textwrap.fill(ai_response, width=100))
                    AssistantFrame.update()
                    speak(ai_response)
                    webbrowser.open("https://www.youtube.com")

                elif 'tell me a joke' in query:
                    ai_response = "Why did the scarecrow win an award? Because he was outstanding in his field!"
                    response_label.config(text=textwrap.fill(ai_response, width=100))
                    AssistantFrame.update()
                    speak(ai_response)

                elif 'tell me the time' in query or 'what is the time' in query:
                    current_time = datetime.datetime.now().strftime("%I:%M %p")
                    ai_response = f"The current time is {current_time}."
                    response_label.config(text=textwrap.fill(ai_response, width=100))
                    AssistantFrame.update()
                    speak(ai_response)

                elif 'tell me a fun fact' in query:
                    ai_response = "Did you know that the word 'robot' comes from the Czech word 'robota', which means 'forced labor'?"
                    response_label.config(text=textwrap.fill(ai_response, width=100))
                    AssistantFrame.update()
                    speak(ai_response)

                elif 'open gmail' in query:
                    ai_response = "Opening Gmail."
                    response_label.config(text=textwrap.fill(ai_response, width=100))
                    AssistantFrame.update()
                    speak(ai_response)
                    webbrowser.open("https://mail.google.com")

                elif 'weather' in query:
                    api_key = "e22091f5fcb17ba19fc9aa2d7c6a3a73"  # replace with your OpenWeatherMap API key
                    base_url = "https://api.openweathermap.org/data/2.5/weather?"
                    ai_response = "Sure, which city's weather would you like to know?"
                    response_label.config(text=textwrap.fill(ai_response, width=100))
                    AssistantFrame.update()
                    speak(ai_response)
                    query = take_command().lower()
                    user_label.config(text=query.capitalize())
                    AssistantFrame.update()
                    city = query
                    complete_url = base_url + "appid=" + api_key + "&q=" + city
                    response = requests.get(complete_url)
                    data = json.loads(response.text)
                    if data['cod'] == '404':
                        ai_response = "Sorry, I could not find the weather information for that city."
                        response_label.config(text=textwrap.fill(ai_response, width=100))
                        AssistantFrame.update()
                        speak(ai_response)
                    else:
                        try:
                            temperature = round(data['main']['temp'] - 273.15)  # convert temperature to Celsius
                            description = data['weather'][0]['description']
                            response_label.config(
                                text=f"The weather in {city.title()} is {temperature} degrees Celsius with {description}.")
                            AssistantFrame.update()
                            speak(f"The weather in {city.title()} is {temperature} degrees Celsius with {description}.")
                        except KeyError:
                            response_label.config(
                                text="Sorry, there was an error retrieving the weather information.")
                            AssistantFrame.update()
                            speak("Sorry, there was an error retrieving the weather information.")
                elif 'bye' in query:
                    response_label.config(text="Goodbye!")
                    AssistantFrame.update()
                    speak("Goodbye!")
                    response_label.config(text="")
                    user_label.config(text="")
                    AssistantFrame.update()
                    exit(0)
                elif 'hello' in query:
                    ai_response = "Hello! How can I assist you?"
                    response_label.config(text=textwrap.fill(ai_response, width=100))
                    AssistantFrame.update()
                    speak(ai_response)
                elif 'thank you' in query:
                    ai_response = "You're Welcome. How can I help you more"
                    response_label.config(text=textwrap.fill(ai_response, width=100))
                    AssistantFrame.update()
                    speak(ai_response)
                elif 'how are you' in query:
                    ai_response = "I'm Very fine. I hope you are good too!"
                    response_label.config(text=textwrap.fill(ai_response, width=100))
                    AssistantFrame.update()
                    speak(ai_response)
                elif 'exit' in query or 'goodbye' in query:
                    user_label.config(text=query)
                    response_label.config(text="Goodbye!")
                    AssistantFrame.update()
                    speak("Goodbye!")
                    response_label.config(text="")
                    user_label.config(text="")
                    AssistantFrame.update()
                    exit(0)
                else:
                    response_label.config(text="Sorry, I didn't understand.\n Please try again or say 'exit' to quit.")
                    AssistantFrame.update()
                    ai_response = "Sorry, I didn't understand. Please try again or say 'exit' to quit. "
                    speak(ai_response)
            # if user input is not "hello" or "exit/goodbye"
            else:
                response_label.config(
                    text="Sorry, I didn't understand.\nPlease start the voice assistant again.")
                speak("Sorry, I didn't understand. Please start the voice assistant again.")
        elif 'exit' in query or 'goodbye' in query:
            user_label.config(text=query)
            response_label.config(text="Goodbye!")
            AssistantFrame.update()
            speak("Goodbye!")
            response_label.config(text="")
            user_label.config(text="")
            AssistantFrame.update()
            exit(0)
        time.sleep(1)


def start_assistant():
    global assistant_running
    assistant_running = True
    response_label.config(text="Hello! Nice to meet you!")
    response_label.update()
    ai_response = "Hello! Nice to meet you!"
    t = threading.Thread(target=speak, args=(ai_response,))
    t.start()
    t = threading.Thread(target=process_input)
    t.start()


def stop_assistant():
    global assistant_running
    assistant_running = False



######################################## USED STUFFS ############################################

def clear1():
    response_label.config(text="")
    user_label.config(text="")
    stop_assistant()
    show_frame(MainPage)


def contact():
    mess._show(title='Contact us', message="Please contact us at : himalpanta@gmail.com ")
    engine.speak("Please Contact us at : himalpanta@gmail.com")


def confirm2():
    answer = askyesno(title='Quit', message='Are you sure that you want to Quit?')
    if answer:
        window.destroy()


def show_frame(frame):
    t = threading.Thread(target=frame.tkraise())
    t.start()
    update_time()
    tick()


def increase_size(event):
    event.widget.config(font=('Quantum Mechanics', 13, 'bold'))


def decrease_size(event):
    event.widget.config(font=('Quantum Mechanics', 12))


def update_time():
    current_time = time.strftime("%I:%M:%S %p")
    clock_label.configure(text=current_time)
    MainPage.after(1000, update_time)


def ComputervisionPage1():
    global camera_initialized
    if not camera_initialized:
        Camera()

    response = " Computer Vision Page Selected "
    t = threading.Thread(target=speak1, args=(response,))
    t.start()
    show_frame(ComputerVisionPage)


def AssistantPage1():
    show_frame(AssistantPage)
    response = " Assistant Page Selected "
    t = threading.Thread(target=speak1, args=(response,))
    t.start()
    response_label.config(text="Press Start Assistance Button ")


def Back_btn():
    show_frame(MainPage)
    response = " Back Button Selected "
    t = threading.Thread(target=speak1, args=(response,))
    t.start()
    update_time()
    tick()


def Back_btn1():
    show_frame(MainPage)
    response = " Back Button Selected "
    t = threading.Thread(target=speak1, args=(response,))
    t.start()
    update_time()
    tick()

def Back_btn2():
    show_frame(ComputerVisionPage)
    response = " Back Button Selected "
    t = threading.Thread(target=speak1, args=(response,))
    t.start()
    update_time()
    tick()

################################ USER INTERFACE DESIGN ##################################

window = tk.Tk()
window.geometry("1366x768")
window.resizable(False, False)
window.title("Envision AI")
window.rowconfigure(0, weight=1)
window.tk.call('wm', 'iconphoto', window._w, tk.PhotoImage(file='icon.png'))

filename1 = ImageTk.PhotoImage(Image.open('images\\Background-Frame-CV.jpg'))

filename2 = ImageTk.PhotoImage(Image.open('images\\Background-Frame-Ai.jpg'))

filename = ImageTk.PhotoImage(Image.open('images\\Background.jpg'))

########################################################--------MAIN MENU PAGE############################################################################

MainPage = tk.Frame(window, width=1366, height=768)
MainPage.grid(row=0, column=0, stick='nsew')
background_label = tk.Label(MainPage, image=filename)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

button1 = Button(MainPage, text="C o m p u t e r  V i s i o n", font=('Quantum Mechanics', 12),
                 command=lambda: ComputervisionPage1(), relief=RAISED, bg="white", fg="black",
                 width=19, height=1, borderwidth=0)
button1.bind("<Enter>", increase_size)
button1.bind("<Leave>", decrease_size)
button1.pack()
button1.place(x=60, y=490)

button2 = Button(MainPage, text="A I  A s s i s t a n t", font=('Quantum Mechanics', 12),
                 command=lambda: AssistantPage1(), relief=RAISED, bg="white", fg="black",
                 width=18, height=1, borderwidth=0)
button2.bind("<Enter>", increase_size)
button2.bind("<Leave>", decrease_size)
button2.pack()
button2.place(x=400, y=490)


button4 = Button(MainPage, text=" Q u i t ", font=('Quantum Mechanics', 11, 'bold'), command=lambda: confirm2(),
                 bg="#211c44", fg="#fff", width=10, height=0, borderwidth=0)
button4.pack()
button4.place(x=633, y=685)

datef = tk.Label(MainPage, text="" + day + "-" + mont[month] + "-" + year + "", fg="black", bg="white", height=1,
                 font=('Century Gothic', 25))
datef.pack()
datef.place(x=80, y=80)

#################################--------------Computer Vision PAGE------------------############################################################
ComputerVisionPage = tk.Frame(window, width=1366, height=768)
ComputerVisionPage.grid(row=0, column=0, stick='nsew')
background_label = tk.Label(ComputerVisionPage, image=filename1)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

CVFrame = tk.Frame(ComputerVisionPage, bg="white")
CVFrame.place(relx=0.2, rely=0.32, relwidth=0.25, relheight=0.50)

CVFrame1 = tk.Frame(ComputerVisionPage, bg="#D0D3D4")
CVFrame1.place(relx=0.53, rely=0.22, relwidth=0.38, relheight=0.75)

head2 = tk.Label(CVFrame, text="               Control Panel                         ", fg="White", bg="#424949",
                 font=('Century Gothic', 17), height=1)
head2.place(x=0, y=0)

head3 = tk.Label(CVFrame1, text="                        Camera Feed Section                            ", fg="White",
                 bg="#424949", font=('Century Gothic', 17), height=1)
head3.place(x=0, y=0)

StartCV = tk.Button(CVFrame, text="➔ Start Computer Vision", command=lambda: computervision1(), fg="white",
                    bg="#004aad",
                    width=24, height=1, activebackground="white", font=('Century Gothic', 15))
StartCV.place(x=15, y=90)

StartFace = tk.Button(CVFrame, text="➔ Face Recognition", command=lambda: [show_frame(SavedPersonsFrame),thumbnail_treeview.update_list(), load_known_faces1()], fg="white",
                    bg="#004aad",
                    width=24, height=1, activebackground="white", font=('Century Gothic', 15))
StartFace.place(x=15, y=150)

clock_label = tk.Label(ComputerVisionPage, fg="white", bg="#004aad", height=1, font=('Century Gothic', 25))
clock_label.pack()
clock_label.place(x=1000, y=80)

update_time()


back = tk.Button(ComputerVisionPage, text="Back", command=lambda: Back_btn(), fg="white", bg="#404040",
                 width=20, height=1, activebackground="white", font=('Century Gothic', 15))
back.place(x=320, y=680)


###############################------------------AI Assistance PAGE--------------------###################################################
AssistantPage = tk.Frame(window, width=1366, height=768)
AssistantPage.grid(row=0, column=0, stick='nsew')
background_label = tk.Label(AssistantPage, image=filename2)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

image = Image.open("images\\Background-Frame-Ai.jpg")
image = image.filter(ImageFilter.GaussianBlur(radius=10))

AssistantFrame = tk.Frame(AssistantPage, bg="White", borderwidth=1)
AssistantFrame.place(relx=0.20, rely=0.2, relwidth=0.60, relheight=0.68)

user_Name = Label(AssistantPage, text="User: ", font=('Century Gothic', 14), bg='white', height=2)
user_Name.pack()
user_Name.place(x=270, y=555)

user_label = Label(AssistantFrame, text="", font=('Century Gothic', 14), fg='black', height=2, width=65, wraplength=500,
                   anchor="w", padx=20)
user_label.pack()
user_label.place(x=50, y=400)

Computer_label = Label(AssistantPage, text="Envision:", font=('Century Gothic', 14), fg='Black', bg='white',
                       height=2)
Computer_label.pack()
Computer_label.place(x=270, y=295)

response_label = Label(AssistantFrame, text="Press Start Assistance Button", font=('Century Gothic', 14), fg='Black',
                       bg='white', height=10, width=100, wraplength=500, anchor="w")
response_label.pack()
response_label.place(x=200, y=50)

StartAI = tk.Button(AssistantPage, text="➔ Start Assistant", command=lambda: start_assistant(), fg="white",
                    bg="#1c1430", width=20, height=1, activebackground="white", font=('Century Gothic', 15),
                    borderwidth=0)
StartAI.place(x=200, y=700)

back = tk.Button(AssistantPage, text="Back", command=lambda: [Back_btn(), clear1()], fg="white", bg="#404040", width=20,
                 height=1, activebackground="white", font=('Century Gothic', 15), borderwidth=0)
back.place(x=950, y=700)

clock = tk.Label(AssistantPage, fg="white", bg="#004aad", height=1, font=('Century Gothic', 25))
clock.pack()
clock.place(x=1000, y=80)

tick()
###############################------------------Saved Personell PAGE--------------------###################################################

# New frame to show all saved persons
SavedPersonsFrame = tk.Frame(window, width=1366, height=768)
SavedPersonsFrame.grid(row=0, column=0, sticky='nsew')

# Background label
background_label2 = tk.Label(SavedPersonsFrame, image=filename2)
background_label2.place(x=0, y=0, relwidth=1, relheight=1)

pplFrame = tk.Frame(SavedPersonsFrame, bg="white")
pplFrame.place(relx=0.2, rely=0.32, relwidth=0.25, relheight=0.50)

head3 = tk.Label(pplFrame, text="               Control Panel                         ", fg="White", bg="#424949",
                 font=('Century Gothic', 17), height=1)
head3.place(x=0, y=0)


class ThumbnailTreeView(ttk.Treeview):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self["columns"] = ("S.N.", "People")
        self.column("#0", width=0, stretch=tk.NO)
        self.column("S.N.", width=50, anchor=tk.CENTER)
        self.column("People", width=200, anchor=tk.CENTER)

        self.heading("S.N.", text="S.N.")
        self.heading("People", text="People")
        self.image_list = []
        self.bind("<Double-1>", self.on_double_click)

    def add_item(self, sn, name, image_path):
        thumbnail_image = self.create_thumbnail(image_path)
        self.image_list.append(thumbnail_image)
        self.insert(parent="", index=tk.END, iid=None, text="", values=(sn, name, ""), image=thumbnail_image)

    def create_thumbnail(self, image_path, size=(100, 100)):
        image = Image.open(image_path)
        image.thumbnail(size)
        thumbnail_image = ImageTk.PhotoImage(image)
        return thumbnail_image

    def get_selected_name(self):
        selected_item = self.selection()[0]
        selected_name = self.item(selected_item)["values"][1]
        return selected_name

    def get_selected_image_path(self):
        selected_item = self.selection()[0]
        selected_name = self.item(selected_item)["values"][1]
        image_path = selected_name + ".jpg"
        return image_path

    def on_double_click(self, event):
        item = self.identify("item", event.x, event.y)
        if item:
            self.selection_set(item)
            self.open_image()

    def open_image(self):
        image_path = self.get_selected_image_path()
        if platform.system() == "Windows":
            os.startfile(image_path)
        elif platform.system() == "Linux":
            subprocess.call(["xdg-open", image_path])
        elif platform.system() == "Darwin":  # macOS
            subprocess.call(["open", image_path])

    def update_list(self):
        for i in self.get_children():
            self.delete(i)
        image_paths = glob.glob("*.jpg")  # Assumes saved images are in the current directory
        for index, image_path in enumerate(image_paths, start=1):
            name = os.path.splitext(os.path.basename(image_path))[0]
            thumbnail_image = self.create_thumbnail(image_path)
            self.image_list.append(thumbnail_image)
            self.insert("", tk.END, values=(index, name, ""), image=thumbnail_image)

thumbnail_treeview = ThumbnailTreeView(SavedPersonsFrame)
thumbnail_treeview.place(relx=0.5, rely=0.23, relwidth=0.45, relheight=0.7)

image_paths = glob.glob("*.jpg")  # Assumes saved images are in the current directory
for i, image_path in enumerate(image_paths):
    name = os.path.splitext(os.path.basename(image_path))[0]
    thumbnail_treeview.add_item(i + 1, name, image_path)

def on_hover(event):
    # Create a popup menu with two options
    popup_menu = tk.Menu(SavedPersonsFrame, tearoff=0)
    popup_menu.configure( font=('Century Gothic', 15), fg="white", bg="#404040")
    popup_menu.add_command(label="Add new face Via Webcam", command=lambda:create_input_window1() )
    popup_menu.add_command(label="Add new face Via File", command=lambda:create_input_window2() )

    # Display the popup menu
    try:
        popup_menu.tk_popup(event.x_root, event.y_root)
    finally:
        popup_menu.bind("<Leave>", lambda _: popup_menu.unpost())

AddFace = tk.Button(pplFrame, text="➔ Add Face", font=('Century Gothic', 15), fg="white", bg="#404040", width=20, height=1)
AddFace.bind("<Enter>", on_hover)
AddFace.place(x=50, y=80)



def create_input_window1():
    input_window = tk.Toplevel()
    input_window.title("Delete Face")
    input_window.configure(bg='light blue')
    input_window.geometry("400x160")
    # Get the screen width and height
    screen_width = input_window.winfo_screenwidth()
    screen_height = input_window.winfo_screenheight()

    # Calculate the x and y coordinates to center the window
    x = (screen_width - input_window.winfo_reqwidth()) // 2
    y = (screen_height - input_window.winfo_reqheight()) // 2

    # Set the window position
    input_window.geometry("+{}+{}".format(x, y))

    name_label = tk.Label(input_window, text="Enter the name of the person to Add:", bg='light blue',font=('Century Gothic', 13 ))
    name_label.place(x=30,y=10)

    name_entry = tk.Entry(input_window,width=25 ,fg="black",relief='solid',font=('Century Gothic', 13))
    name_entry.place(x=70, y=70)

    def submit():
        name = name_entry.get()
        add_new_face(name)
        input_window.destroy()

    submit_button = tk.Button(input_window, text="Submit", command=submit, fg="White"  ,bg="#111211" ,height=1,width=20 , activebackground = "white" ,font=('Century Gothic', 10))
    submit_button.place(x=205, y=120)

    cancel_button = tk.Button(input_window, text="Cancel", command=input_window.destroy, fg="White", bg="#614f41", height = 1,width=20, activebackground="white", font=('Century Gothic', 10))
    cancel_button.place(x=20, y=120)
    thumbnail_treeview.update_list()
    load_known_faces1()

def create_input_window2():
    input_window = tk.Toplevel()
    input_window.title("Delete Face")
    input_window.configure(bg='light blue')
    input_window.geometry("400x160")
    # Get the screen width and height
    screen_width = input_window.winfo_screenwidth()
    screen_height = input_window.winfo_screenheight()

    # Calculate the x and y coordinates to center the window
    x = (screen_width - input_window.winfo_reqwidth()) // 2
    y = (screen_height - input_window.winfo_reqheight()) // 2

    # Set the window position
    input_window.geometry("+{}+{}".format(x, y))

    name_label = tk.Label(input_window, text="Enter the name of the person to Add:", bg='light blue',font=('Century Gothic', 13 ))
    name_label.place(x=30,y=10)

    name_entry = tk.Entry(input_window,width=25 ,fg="black",relief='solid',font=('Century Gothic', 13))
    name_entry.place(x=70, y=70)

    def submit():
        name = name_entry.get()
        add_new_face_from_image(name)
        input_window.destroy()

    submit_button = tk.Button(input_window, text="Submit", command=submit, fg="White"  ,bg="#111211" ,height=1,width=20 , activebackground = "white" ,font=('Century Gothic', 10))
    submit_button.place(x=205, y=120)

    cancel_button = tk.Button(input_window, text="Cancel", command=input_window.destroy, fg="White", bg="#614f41", height = 1,width=20, activebackground="white", font=('Century Gothic', 10))
    cancel_button.place(x=20, y=120)
    thumbnail_treeview.update_list()
    load_known_faces1()

update_button = tk.Button(pplFrame, text="Update", command=lambda :[update_known_face_button(thumbnail_treeview.get_selected_name()), thumbnail_treeview.update_list(), load_known_faces1()], font=('Century Gothic', 15), fg="white", bg="#404040", width=20, height=1)
update_button.place(x=50, y=160)

delete_button = tk.Button(pplFrame, text="Delete", command=lambda: [delete_face(thumbnail_treeview.get_selected_name()), thumbnail_treeview.update_list(), load_known_faces1()], font=('Century Gothic', 15), fg="white", bg="#404040", width=20, height=1)
delete_button.place(x=50, y=230)

# Back button
back2 = tk.Button(SavedPersonsFrame, text="Back", command=lambda:Back_btn2(), font=('Century Gothic', 15), fg="white", bg="#404040", width=20, height=1)
back2.place(relx=0.1, rely=0.9)


########################################################################################################################
show_frame(ComputerVisionPage)

window.mainloop()
