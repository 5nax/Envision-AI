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
import cv2
import pyttsx3
import requests
import speech_recognition as sr
import wikipedia
from PIL import Image, ImageFilter
from PIL import ImageTk
import random
import os
from tkinter import simpledialog, messagebox
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
###########################################################################################

language = 'en'

engine = pyttsx3.init('sapi5')

voices = engine.getProperty('voices')

engine.setProperty('voice', voices[1].id)

rate = engine.getProperty('rate')

engine.setProperty('rate', 130)

assistant_running = False

running = True

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
        user_label1.config(text="Listening...")
        user_label1.update()
        print("Listening...")
        r.pause_threshold = 1
        audio = r.listen(source)
    try:
        print("Recognizing...")
        user_label.config(text="Recognizing...")
        user_label.update()
        user_label1.config(text="Recognizing...")
        user_label1.update()
        query = r.recognize_google(audio, language='en-us')
        print(f"User said: {query}\n")
    except Exception as e:
        print(e)
        return "Speech Recognition could not understand audio"
    return query


####################################### Computer Vision Person Register ####################################################

def add_new_face():

    # Get the name for the new face
    name = simpledialog.askstring("Input", "Enter the name of the person:",
                                   parent=ComputerVisionPage)
    if name:
        # Capture the image and save it with the given name
        capture_image_and_save(name)

        # Encode the face and update the known faces
        new_face_encoding, _ = encode_faces([f"{name}.jpg"], [name])
        known_face_encodings, known_face_names = load_known_faces("known_faces.pkl")
        known_face_encodings.extend(new_face_encoding)
        known_face_names.extend([name])

        # Save the updated known faces
        save_known_faces(known_face_encodings, known_face_names, "known_faces.pkl")
        messagebox.showinfo("Success", f"{name} has been added successfully!")


def capture_image_and_save(name):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        cv2.imshow("Press 's' to save, 'q' to quit", frame)
        key = cv2.waitKey(1)

        if key == ord('s'):
            cv2.imwrite(f"{name}.jpg", frame)
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def add_new_face_from_image():
    # Get the name for the new face
    name = simpledialog.askstring("Input", "Enter the name of the person:",
                                   parent=ComputerVisionPage)
    if name:
        # Open a file dialog to select the image
        image_path = filedialog.askopenfilename(title="Select the image of the person",
                                                filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

        # Check if an image is selected
        if not image_path:
            messagebox.showerror("Error", "No image selected.")
            return

        # Save a copy of the image with the given name
        shutil.copy(image_path, f"{name}.jpg")

        # Encode the face and update the known faces
        new_face_encoding, _ = encode_faces([f"{name}.jpg"], [name])
        known_face_encodings, known_face_names = load_known_faces("known_faces.pkl")
        known_face_encodings.extend(new_face_encoding)
        known_face_names.extend([name])

        # Save the updated known faces
        save_known_faces(known_face_encodings, known_face_names, "known_faces.pkl")
        messagebox.showinfo("Success", f"{name} has been added successfully!")

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


known_face_encodings, known_face_names = load_known_faces("known_faces.pkl")


def update_known_face(name, known_face_encodings, known_face_names, filename):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Press 's' to save the current frame or 'q' to quit", frame)
        key = cv2.waitKey(1)

        if key == ord('s'):
            # Save the current frame as a temporary image
            cv2.imwrite('temp_image.jpg', frame)
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Encode the face from the temporary image
    temp_image_path = 'temp_image.jpg'
    new_face_encoding = face_recognition.face_encodings(face_recognition.load_image_file(temp_image_path))[0]

    # Find the index of the existing face encoding with the provided name
    face_index = known_face_names.index(name)

    # Replace the existing encoding and name in the lists
    known_face_encodings[face_index] = new_face_encoding

    # Save the updated data to the file
    save_known_faces(known_face_encodings, known_face_names, filename)

    # Remove the temporary image file
    os.remove(temp_image_path)
    messagebox.showinfo("Success", f"{name} has been updated successfully!")


def update_known_face_button():
    known_face_encodings, known_face_names
    name = simpledialog.askstring("Update Known Face", "Enter the name of the person whose image you want to update:")
    if name is not None:
        if name in known_face_names:
            update_known_face(name, known_face_encodings, known_face_names, "known_faces.pkl")
            messagebox.showinfo("Update Known Face", f"Updated the image for {name}.")
        else:
            messagebox.showerror("Update Known Face", f"No saved image found for {name}.")


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
def face_detection_thread(face_detector, known_face_encodings, known_face_names, frame_queue, face_info_queue, exit_event):
    while not exit_event.is_set():
        frame = frame_queue.get()
        face_locations, face_names = detect_known_faces(face_detector, known_face_encodings, known_face_names, frame)
        face_info_queue.put((face_locations, face_names))

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


########################################################################################

def Game1():
    response = "Start Game Selected"
    speak(response)
    t1 = threading.Thread(target=Game)
    t1.start()


def display_text_and_speak(text):
    strlabel.config(text=text)
    strlabel.update()
    speak(text)


def ask_question_and_get_response(question):
    display_text_and_speak(question)
    while True:
        response = take_command().lower()
        user_label1.config(text=response)
        if response in ["yes", "no", "exit"]:
            break
        else:
            display_text_and_speak("I didn't catch that. Please say 'yes' or 'no'.")
    return response


def Game():
    # display main menu
    display_text_and_speak("Welcome to Get to the University Game!")
    display_text_and_speak("The game where you try to get in on time to your lessons!")

    def get_user_input():
        while True:
            play = ask_question_and_get_response("Would you like to play?")
            if play == "yes":
                display_text_and_speak("Let's go!")
                time.sleep(1)
                gamerun()
                again = ask_question_and_get_response("Do you want to play again?")
                if again == "no":
                    display_text_and_speak("Thanks for playing!")
                    break
            elif play == "no" or play == "exit":
                display_text_and_speak("Sure! Quitting the game")
                stop_function()
                strlabel.config(text="Press Start Game Button")
                strlabel.update()
                return

        # Reset the GUI labels
        user_label1.config(text="")
        strlabel.config(text="")
        strlabel.update()
        user_label1.update()

    threading.Thread(target=get_user_input).start()


def gamerun():
    global running
    if running:
        # set initial values for variables

        strlabel.config(text="After a busy night at the local bar you wake up at 8:30...")
        strlabel.update()
        speak("After a busy night at the local bar you wake up at 8:30...")
        strlabel.config(text="But.. YOUR COLLEGE IS AT 9 AM !!")
        strlabel.update()
        speak("But.. YOUR COLLEGE IS AT 9 AM !!")

        strlabel.config(text="You must get to college on time, lets hope nothing slows you down...")
        strlabel.update()
        speak("You must get to college on time, lets hope nothing slows you down...")

    while True:
        totaltime = 0
        timetaken = 0
        count = 0
        inv = []
        used = []

        strlabel.config(text="Say Start to start game or exit to stop")
        strlabel.update()
        speak("say continue to start game or exit to stop")
        response = take_command().lower()
        if response == "exit":
            speak("Exiting game.")
            break
        elif response != "start":
            speak("Invalid response. Please say 'continue' or 'exit'.")
            continue

        used = []
        totaltime = 0
        for count in range(0, 100):
            used.append(count)
        count = 0
        # repeats 3 times so 3 events happen every game
        while count < 3:
            count += 1
            choice = random.randint(0, 99)
            # if number has already been generated then it will not run it and will change
            # that value to 101 which is my value for its not a valid choice
            if used[choice] != 101:
                used[choice] = 101
                userpl = True
                # random Choice for selecting the path
                if choice == 1:
                    timetaken = Money()
                elif choice == 2:
                    timetaken = river()
                elif choice == 3:
                    timetaken = Kitten()
                elif choice == 4:
                    timetaken = prisoner()
                elif choice == 5:
                    timetaken = dinosaur()
                elif choice == 6:
                    timetaken = lost_child()
                elif choice == 7:
                    timetaken = AppleStore()
                elif choice == 8:
                    timetaken = kidnapping()
                else:
                    userpl = False
                    count -= 1
                # when an event is run then the user must hit enter to continue
                if userpl:
                    strlabel.config(text="Say continue to proceed")
                    strlabel.update()
                    speak("Say continue to proceed")
                    a = take_command().lower()
                    user_label1.update()
                    if a == "exit":
                        speak("Exiting game.")
                        return

            else:
                # will not count that as one of the 3 options
                used[choice] = 101
                timetaken = 0
                count -= 1
            # sets the user played value to true which only gets changed
            # when the user has a go
            userpl = True
            totaltime = totaltime + timetaken
        if (totaltime) <= 0:
            strlabel.config(text="You've finally made it to the university without being Late \n YOU WIN!!!!")
            strlabel.update()
            speak("You've finally made it to the university without being Late........... YOU WIN!!!!")
        else:
            strlabel.config(
                text="You've finally made it to the university, and you were only {0} minutes late!".format(totaltime))
            strlabel.update()
            speak("You've finally made it to university, and you were {0} minutes late!".format(totaltime))


def stop_function():
    global running
    running = False
    strlabel.config(text=" ")
    user_label1.config(text=" ")
    show_frame(MainPage)
    return



def vinput():
    # validates inputs so it can't crash and the choice can only be a, b, or c
    valid_inputs = ["number one", "number 2", "number 3"]

    while True:
        inp = take_command().lower()
        if inp in valid_inputs:
            break
        else:
            speak("That's not a valid input")

    return inp


def Money():
    global timetaken
    strlabel.config(text="You're on your way to university when you stumble upon a lost wallet on the ground.")
    strlabel.update()
    speak("You're on your way to university when you stumble upon a lost wallet on the ground.")
    strlabel.config(text="You pick it up and see that it contains a large sum of money and an ID card.")
    strlabel.update()
    speak("You pick it up and see that it contains a large sum of money and an ID card.")
    strlabel.config(text="What do you do with the wallet and its contents?")
    strlabel.update()
    speak("What do you do with the wallet and its contents?")
    strlabel.config(text="Number 1: Take the money and ditch the wallet.")
    strlabel.update()
    speak("Number 1: Take the money and ditch the wallet.")
    strlabel.config(text="Number 2: Try to locate the owner using the ID card.")
    strlabel.update()
    speak("Number 2: Try to locate the owner using the ID card.")
    strlabel.config(text="Number 3: Take the wallet and money to the police station.")
    strlabel.update()
    speak("Number 3: Take the wallet and money to the police station.")

    inp = vinput()
    user_label1.config(text=inp)
    user_label1.update()
    if inp == "number one":
        strlabel.config(text="You've just committed theft. Shame on you!")
        strlabel.update()
        speak("You've just committed theft. Shame on you!")
        timetaken = 0
    elif inp == "number 2":
        strlabel.config(text="You manage to contact the owner and return the wallet with all the money intact.")
        strlabel.update()
        speak("You manage to contact the owner and return the wallet with all the money intact.")
        strlabel.config(text="They're extremely grateful and offer to buy you lunch as a thank you.")
        strlabel.update()
        speak("They're extremely grateful and offer to buy you lunch as a thank you.")
        timetaken = 10
    elif inp == "number 3":
        strlabel.config(text="You take the wallet to the police station and they're able to locate the owner.")
        strlabel.update()
        speak("You take the wallet to the police station and they're able to locate the owner.")
        strlabel.config(text="They thank you for your honesty and give you a certificate of recognition.")
        strlabel.update()
        speak("They thank you for your honesty and give you a certificate of recognition.")
        timetaken = 15
    return timetaken


def river():
    global timetaken
    strlabel.config(text="You come across a wide river blocking your path.")
    strlabel.update()
    speak("You come across a wide river blocking your path.")
    strlabel.config(text="What will you do?")
    strlabel.update()
    speak("What will you do?")
    strlabel.config(text="Number 1: Try to swim across.")
    strlabel.update()
    speak("Number 1: Try to swim across.")
    strlabel.config(text="Number 2: Steal the boat in front of you.")
    strlabel.update()
    speak("Number 2: Steal the boat in front of you")
    strlabel.config(text="Number 3: Search for a bridge or a boat to cross.")
    strlabel.update()
    speak("Number 3: Search for a bridge or a boat to cross.")

    inp = vinput()
    user_label1.config(text=inp)
    user_label1.update()
    if inp == "number one":
        strlabel.config(text="You attempt to swim across, but the current is too strong and you are swept downstream.")
        strlabel.update()
        speak("You attempt to swim across, but the current is too strong and you are swept downstream.")
        #
        timetaken = 25
    elif inp == "number two":
        strlabel.config(
            text="You Managed to steal the boat and cross the river in time \n but the police will be looking for you! ")
        strlabel.update()
        speak("You Managed to steal the boat and cross the river in time but the police will be looking for you!")
        #
        timetaken = 0
    elif inp == "number three":
        strlabel.config(text="You spend some time searching and eventually find a small boat to cross the river.")
        strlabel.update()
        speak("You spend some time searching and eventually find a small boat to cross the river.")
        #
        timetaken = 20
    return timetaken


def AppleStore():
    global timetaken, timetaken
    strlabel.config(text="As you approach the AppleStore, you hear loud noises and see a commotion.")
    strlabel.update()
    speak("As you approach the AppleStore, you hear loud noises and see a commotion.")
    strlabel.config(text="You notice people breaking into the store and taking iPads, iPhones, and Mac-books.")
    strlabel.update()
    speak("You notice people breaking into the store and taking iPads, iPhones, and Mac-books.")
    strlabel.config(text="What do you do?")
    strlabel.update()
    speak("What do you do?")
    strlabel.config(text="Number 1: Call the police.")
    strlabel.update()
    speak("Number 1: Call the police.")
    strlabel.config(text="Number 2: Try to stop the looters and protect the store.")
    strlabel.update()
    speak("Number 2:Try to stop the looters and protect the store.")
    strlabel.config(text="Number 3: Walk away and continue on your way to work")
    strlabel.update()
    speak("Number 3: Walk away and continue on your way to work.")

    inp = vinput()
    user_label1.config(text=inp)
    if inp == "number one":
        strlabel.config(text="You quickly dial 911 and report the situation.")
        strlabel.update()
        speak("You quickly dial 911 and report the situation.")
        strlabel.config(text="The police arrive and handle the situation, but you're slightly delayed.")
        strlabel.update()
        speak("The police arrive and handle the situation, but you're slightly delayed.")
        timetaken = 5
    elif inp == "number 2":
        strlabel.config(text="You try to stop the looters and protect the store, but it's too dangerous.")
        strlabel.update()
        speak("You try to stop the looters and protect the store, but it's too dangerous.")
        strlabel.config(text="You get caught up in the chaos and barely escape with your life.")
        strlabel.update()
        speak("You get caught up in the chaos and barely escape with your life.")
        strlabel.config(text="You're late for work and have to explain what happened to your boss.")
        strlabel.update()
        speak("You're late for work and have to explain what happened to your boss.")
        timetaken = 25
    elif inp == "number 3":
        strlabel.config(text="You decide to play it safe and walk away from the situation.")
        strlabel.update()
        speak("You decide to play it safe and walk away from the situation.")
        strlabel.config(text="You make it to work on time and avoid any trouble.")
        strlabel.update()
        speak("You make it to work on time and avoid any trouble.")
        timetaken = 0
    return timetaken


def lost_child():
    global timetaken
    strlabel.config(text="You are walking in the park when you notice a lost child crying")
    strlabel.update()
    speak("You are walking in the park when you notice a lost child crying")
    strlabel.config(text="What do you do?")
    strlabel.update()
    speak("What do you do?")
    strlabel.config(text="Number 1: Look around for the child's parents and try to reunite them")
    strlabel.update()
    speak("Number 1: Look around for the child's parents and try to reunite them")
    strlabel.config(text="Number 2: Stay with the child until the police arrive")
    strlabel.update()
    speak("Number 2: Stay with the child until the police arrive")
    strlabel.config(text="Number 3: Ignore the child and continue on your way")
    strlabel.update()
    speak("Number 3: Ignore the child and continue on your way")
    inp = vinput()
    user_label1.config(text=inp)
    user_label1.update()
    if inp == "number one":
        strlabel.config(text="Good job! You helped reunite the lost child with their parents")
        strlabel.update()
        speak("Good job! You helped reunite the lost child with their parents")
        timetaken = 10
    elif inp == "number 2":
        strlabel.config(text="You stayed with the child until the police arrived, and they thanked you for your help")
        strlabel.update()
        speak("You stayed with the child until the police arrived, and they thanked you for your help")
        timetaken = 20
    elif inp == "number 3":
        strlabel.config(text="Shame on you! You should always help someone in need")
        strlabel.update()
        speak("Shame on you! You should always help someone in need")
        timetaken = 0
    return timetaken


def kidnapping():
    strlabel.config(text="You're walking to university when you suddenly hear footsteps behind you")
    strlabel.update()
    speak("You're walking to university when you suddenly hear footsteps behind you.")
    strlabel.config(text="As you turn around, a group of strangers grab you and put a bag over your head.")
    strlabel.update()
    speak("As you turn around, a group of strangers grab you and put a bag over your head.")
    strlabel.config(text="You can hear them driving for what seems like hours until they finally stop.")
    strlabel.update()
    speak("You can hear them driving for what seems like hours until they finally stop.")
    strlabel.config(text="They pull you out of the car and throw you into a room.")
    strlabel.update()
    speak("They pull you out of the car and throw you into a room.")
    strlabel.config(text="You have no idea where you are or why this is happening.")
    strlabel.update()
    speak("You have no idea where you are or why this is happening.")
    strlabel.config(text="So, what do you think happened?")
    strlabel.update()
    speak("So, what do you think happened?")
    strlabel.config(text="Number 1: You got kidnapped.")
    strlabel.update()
    speak("Number 1: You got kidnapped.")
    strlabel.config(text="Number 2: You got kidnapped.")
    strlabel.update()
    speak("Number 2: You got kidnapped.")
    strlabel.config(text="Number 3: You got kidnapped.")
    strlabel.update()
    speak("Number 3: You got kidnapped.")
    inp = vinput()
    user_label1.config(text=inp)
    user_label1.update()
    strlabel.config(text="Wow! Now wasn't that hard to figure out!")
    strlabel.update()
    speak("Wow! Now wasn't that hard to figure out!")
    strlabel.config(text="Looks like you have nothing to do about it, too...")
    strlabel.update()
    speak("Looks like you have nothing to do about it, too...")
    strlabel.config(text="You spend the next few days in that room, trying to find a way out.")
    strlabel.update()
    speak("You spend the next few days in that room, trying to find a way out.")
    strlabel.config(text="Finally, you hear someone opening the door.")
    strlabel.update()
    speak("Finally, you hear someone opening the door.")
    strlabel.config(text="It's the police! They found you just in time.")
    strlabel.update()
    speak("It's the police! They found you just in time.")
    strlabel.config(text="You're safe now, but shaken up.")
    strlabel.update()
    speak("You're safe now, but shaken up.")
    strlabel.config(text="Let's keep going!... Better late than never...i guess")
    strlabel.update()
    speak("Let's keep going!... Better late than never...i guess")
    timetaken = 4320
    return timetaken


def Kitten():
    global timetaken
    strlabel.config(text="As you walk, you hear a soft meowing sound coming from behind a nearby tree.")
    strlabel.update()
    speak("As you walk, you hear a soft meowing sound coming from behind a nearby tree.")
    strlabel.config(text="Upon closer inspection, you find a small, lost kitten.")
    strlabel.update()
    speak("Upon closer inspection, you find a small, lost kitten.")
    strlabel.config(text="What would you like to do with the kitten?")
    strlabel.update()
    speak("What would you like to do with the kitten?")
    strlabel.config(text="Number 1: Take the kitten with you and try to find its owner.")
    strlabel.update()
    speak("Number 1: Take the kitten with you and try to find its owner.")
    strlabel.config(text="Number 2: Leave the kitten there and continue on your way to university.")
    strlabel.update()
    speak("Number 2: Leave the kitten there and continue on your way to university.")
    strlabel.config(text="Number 3: Take the kitten with you and try to find it a new home.")
    strlabel.update()
    speak("Number 3: Take the kitten with you and try to find it a new home.")
    inp = vinput()
    user_label1.config(text=inp)
    user_label1.update()
    if inp == "number one":
        strlabel.config(text="You're doing the right thing! The kitten's owner will be grateful.")
        strlabel.update()
        speak("You're doing the right thing! The kitten's owner will be grateful.")
        timetaken = 5
    elif inp == "number 2":
        strlabel.config(text="It's unfortunate, but you have to get to university.")
        strlabel.update()
        speak("It's unfortunate, but you have to get to university.")
        timetaken = 0
    elif inp == "number 3":
        strlabel.config(text="You have a big heart, helping the kitten find a new home.")
        strlabel.update()
        speak("You have a big heart, helping the kitten find a new home.")
        timetaken = 15
        strlabel.config(text="Good luck with your day at university!")
        strlabel.update()
        speak("Good luck with your day at university!")
    return timetaken


def prisoner():
    global timetaken
    strlabel.config(text="As you walk down the street, you notice a group of people walking towards you.")
    strlabel.update()
    speak("As you walk down the street, you notice a group of people walking towards you.")
    strlabel.config(text="As they get closer, you realize they are three individuals who look a bit rough.")
    strlabel.update()
    speak("As they get closer, you realize they are three individuals who look a bit rough.")
    strlabel.config(text="They approach you and demand your belongings.")
    strlabel.update()
    speak("They approach you and demand your belongings.")
    strlabel.config(text="What do you do?")
    strlabel.update()
    speak("What do you do?")
    strlabel.config(text="Number 1: Try to de-escalate the situation and talk to them calmly.")
    strlabel.update()
    speak("Number 1: Try to de-escalate the situation and talk to them calmly.")
    strlabel.config(text="Number 2: Hand over your belongings and hope they don't harm you.")
    strlabel.update()
    speak("Number 2: Hand over your belongings and hope they don't harm you.")
    strlabel.config(text="Number 3: Make a run for it and call the police")
    strlabel.update()
    speak("Number 3: Make a run for it and call the police.")

    inp = vinput()
    user_label1.config(text=inp)
    user_label1.update()
    if inp == "number one":
        strlabel.config(text="You were able to calm them down and they let you go.")
        strlabel.update()
        speak("You were able to calm them down and they let you go.")
        #
        timetaken = 5
    elif inp == "number 2":
        strlabel.config(text="You gave them your belongings, but they didn't harm you.")
        strlabel.update()
        speak("You gave them your belongings, but they didn't harm you.")
        #
        timetaken = 7
    elif inp == "number 3":
        strlabel.config(text="You managed to escape and called the police.")
        strlabel.update()
        speak("You managed to escape and called the police.")
        #
        timetaken = 0
    return timetaken


def dinosaur():
    timetaken = 0
    strlabel.config(
        text="You are on your way and come across a drugged grizzly bear \n which escaped from the local zoo.")
    strlabel.update()
    speak("You are on your way and come across a drugged grizzly bear which escaped from the local zoo.")
    strlabel.config(text="It stands on its hind legs and growls at you menacingly.")
    strlabel.update()
    speak("It stands on its hind legs and growls at you menacingly.")
    strlabel.config(text="Quick! What do you want to do?")
    strlabel.update()
    speak("Quick! What do you want to do?")
    strlabel.config(text="Number 1: Slowly back away and try to retreat.")
    strlabel.update()
    speak("Number 1: Slowly back away and try to retreat.")
    strlabel.config(text="Number 2: Make yourself look bigger and yell to scare it off.")
    strlabel.update()
    speak("Number 2: Make yourself look bigger and yell to scare it off.")
    strlabel.config(text="Number 3: Play dead and hope the bear loses interest.")
    strlabel.update()
    speak("Number 3: Play dead and hope the bear loses interest.")
    inp = vinput()
    user_label1.config(text=inp)
    user_label1.update()
    if inp == "number one":
        strlabel.config(text="Good choice. Slowly back away and avoid eye contact.")
        strlabel.update()
        speak("Good choice. Slowly back away and avoid eye contact.")
        strlabel.config(text="The bear loses interest and goes back to foraging.")
        strlabel.update()
        speak("The bear loses interest and goes back to foraging.")
        timetaken = 2
    elif inp == "number two":
        strlabel.config(text="You try to make yourself look bigger and yell loudly.")
        strlabel.update()
        speak("You try to make yourself look bigger and yell loudly.")
        strlabel.config(text="The bear seems startled and charges at you.")
        strlabel.update()
        speak("The bear seems startled and charges at you.")
        strlabel.config(text="You quickly move out of the way and it runs off.")
        strlabel.update()
        speak("You quickly move out of the way and it runs off.")
        timetaken = 0
    elif inp == "number three":
        strlabel.config(text="Playing dead might work, but it's risky.")
        strlabel.update()
        speak("Playing dead might work, but it's risky.")
        strlabel.config(text="The bear sniffs around you and nudges you with its nose.")
        strlabel.update()
        speak("The bear sniffs around you and nudges you with its nose.")
        strlabel.config(text="You hold your breath and stay still until it leaves.")
        strlabel.update()
        speak("You hold your breath and stay still until it leaves.")
        timetaken = 15
        strlabel.config(text="You survived the encounter!")
        strlabel.update()
    speak("You survived the encounter!")
    return timetaken


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


def GamePage1():
    show_frame(GamePage)
    response = " Game Page Selected "
    t = threading.Thread(target=speak1, args=(response,))
    t.start()
    strlabel.config(text="Press Start Game Button ")
    user_label1.config(text="")


def ComputervisionPage1():
    response = " Computer Vision Page Selected "
    t = threading.Thread(target=speak1, args=(response,))
    t.start()
    show_frame(ComputerVisionPage)
    t1 = threading.Thread(target=Camera)
    t1.start()


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
    stop_function()
    update_time()
    tick()


################################ USER INTERFACE DESIGN ##################################

window = tk.Tk()
window.geometry("1366x768")
window.resizable(False, False)
window.title("Envision AI")
window.rowconfigure(0, weight=1)
window.tk.call('wm', 'iconphoto', window._w, tk.PhotoImage(file='icon.png'))

filename1 = ImageTk.PhotoImage(Image.open('Background-Frame-CV.jpg'))

filename2 = ImageTk.PhotoImage(Image.open('Background-Frame-Ai.jpg'))

filename3 = ImageTk.PhotoImage(Image.open('Background-Frame-GG.jpg'))

filename = ImageTk.PhotoImage(Image.open('Background.jpg'))

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
button2.place(x=315, y=490)

button3 = Button(MainPage, text="G a m e", font=('Quantum Mechanics', 12),
                 command=lambda: GamePage1(), bg="white", fg="black",
                 width=17, height=1, borderwidth=0)
button3.bind("<Enter>", increase_size)
button3.bind("<Leave>", decrease_size)
button3.pack()
button3.place(x=600, y=490)

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

clock_label = tk.Label(ComputerVisionPage, fg="white", bg="#004aad", height=1, font=('Century Gothic', 25))
clock_label.pack()
clock_label.place(x=1000, y=80)

update_time()
def on_hover(event):
    # Create a popup menu with two options
    popup_menu = tk.Menu(ComputerVisionPage, tearoff=0)
    popup_menu.configure(bg="#004aad", fg="white", font=('Century Gothic', 15))
    popup_menu.add_command(label="Add new face Via Webcam", command=add_new_face )
    popup_menu.add_command(label="Add new face Via File", command=add_new_face_from_image)

    # Display the popup menu
    try:
        popup_menu.tk_popup(event.x_root, event.y_root)
    finally:
        popup_menu.bind("<Leave>", lambda _: popup_menu.unpost())


AddFace = tk.Button(CVFrame, text="➔ Add Face", fg="white", bg="#004aad",command=None,
                    width=24, height=1, activebackground="white", font=('Century Gothic', 15))
AddFace.place(x=15, y=150)
AddFace.bind("<Enter>", on_hover)

DeleteFace = tk.Button(CVFrame, text="➔ Delete Face", command=lambda: delete_face(simpledialog.askstring("Input", "Enter the name of the person to delete:",
                                   parent=ComputerVisionPage)), fg="white", bg="#004aad",
                    width=24, height=1, activebackground="white", font=('Century Gothic', 15))
DeleteFace.place(x=15, y=270)

UpdateFace = tk.Button(CVFrame, text="➔ Update Known Face", command=update_known_face_button, fg="white", bg="#004aad",
                       width=24, height=1, activebackground="white", font=('Century Gothic', 15))
UpdateFace.place(x=15, y=210)

delete_all_known_faces_button = tk.Button(
    ComputerVisionPage, text="➔ Delete All Known Faces", command=lambda: delete_all_known_faces("known_faces.pkl"),
    fg="white", bg="black", width=24, height=1, activebackground="white", font=('Century Gothic', 15)
)
delete_all_known_faces_button.place(x=15, y=680)

back = tk.Button(ComputerVisionPage, text="Back", command=lambda: Back_btn(), fg="white", bg="#404040",
                 width=20, height=1, activebackground="white", font=('Century Gothic', 15))
back.place(x=320, y=680)

###############################------------------AI Assistance PAGE--------------------###################################################
AssistantPage = tk.Frame(window, width=1366, height=768)
AssistantPage.grid(row=0, column=0, stick='nsew')
background_label = tk.Label(AssistantPage, image=filename2)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

image = Image.open("Background-Frame-Ai.jpg")
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
################################################ Game PAGE ########################################################################

GamePage = tk.Frame(window, width=1366, height=768)
GamePage.grid(row=0, column=0, stick='nsew')

background_label = tk.Label(GamePage, image=filename3)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

gameframe = tk.Frame(GamePage, bg="white")
gameframe.place(relx=0.20, rely=0.2, relwidth=0.60, relheight=0.68)

user_Name = Label(GamePage, text="User: ", font=('Century Gothic', 14), bg='white', height=2)
user_Name.pack()
user_Name.place(x=270, y=555)

user_label1 = Label(gameframe, text="", font=('Century Gothic', 14), fg='black', height=2, width=65, wraplength=500,
                    anchor="w", padx=20)
user_label1.pack()
user_label1.place(x=50, y=400)

strlabel = Label(gameframe, text="Press Start Game Button", font=('Century Gothic', 14), fg='Black',
                 bg='white', height=10, width=100, wraplength=500, anchor="w")
strlabel.pack()
strlabel.place(x=200, y=50)

Computer_label = Label(GamePage, text="Envision:", font=('Century Gothic', 14), fg='Black', bg='white',
                       height=2)
Computer_label.pack()
Computer_label.place(x=270, y=295)

StartGame = tk.Button(GamePage, text="➔ Start Game", command=lambda: Game1(), fg="white",
                      bg="#1c1430", width=20, height=1, activebackground="white", font=('Century Gothic', 15),
                      borderwidth=0)
StartGame.place(x=200, y=700)

clock = tk.Label(GamePage, fg="white", bg="#004aad", height=1, font=('Century Gothic', 25))
clock.pack()
clock.place(x=1000, y=80)
tick()

back = tk.Button(GamePage, text="Back", command=lambda: Back_btn1(), fg="white", bg="#404040", width=20,
                 height=1, activebackground="white", font=('Century Gothic', 15), borderwidth=0)
back.place(x=950, y=700)

########################################################################################################################
show_frame(MainPage)

window.mainloop()
