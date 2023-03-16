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
import numpy as np
import pyttsx3
import requests
import speech_recognition as sr
import wikipedia
from PIL import Image, ImageFilter
from PIL import ImageTk
import random

###########################################################################################

language = 'en'

engine = pyttsx3.init('sapi5')

voices = engine.getProperty('voices')

engine.setProperty('voice', voices[1].id)

rate = engine.getProperty('rate')

engine.setProperty('rate', 150)

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

def speak_label(label, direction, engine):
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

###########################################################################################

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

def run_computer_vision():
    lmain = tk.Label(CVFrame1, text="Computer Vision Started \n Press the key 'Q' on your keyboard to stop\n  Computer Vision",
                     font=('Century Gothic', 14), bg="#D0D3D4",height=30,width=50)
    lmain.place(relx=0, rely=0)
    # Initialize YOLOv3
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # Load classes
    with open("yolov3.txt", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Get output layers
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Initialize camera
    cap = cv2.VideoCapture(0)

    # Initialize frame counter
    frame_count = 0

    while True:
        # Read from camera
        ret, frame = cap.read()
        if not ret:
            break

        # Increment frame counter
        frame_count += 1

        if frame_count % 2 == 0:
            # Downsample frame
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        else:
            # Continue to next frame without processing
            continue

        # Prepare input for YOLOv3
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, False)
        net.setInput(blob)

        # Get outputs from YOLOv3
        outs = net.forward(output_layers)

        # Process outputs
        boxes = []
        confidences = []
        class_ids = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0:
                    cx = int(detection[0] * frame.shape[1])
                    cy = int(detection[1] * frame.shape[0])
                    w = int(detection[2] * frame.shape[1])
                    h = int(detection[3] * frame.shape[0])
                    x = int(cx - w / 2)
                    y = int(cy - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-max suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Draw boxes and labels
        indexes = np.array(indexes)
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            direction = get_direction(x, y, w, h, frame.shape[1], frame.shape[0])
            t = threading.Timer(1, speak_label, args=(label, direction, engine))
            t.start()
        # Show the resulting frame
        cv2.imshow("Detection",frame)
        # Check for key presses
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('image.jpg', frame)

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()
    Camera()

def computervision1():
    global running
    running = True
    t = threading.Thread(target=run_computer_vision)
    t.start()
    speak("Starting Computer Vision")

def Camera():


    head3 = tk.Label(CVFrame1, text="                        Camera Feed Section                            ",
                     fg="White",
                     bg="#424949", font=('Century Gothic', 17), height=1)
    head3.place(x=0, y=0)

    # Capture from camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        # Create a label in the frame to display "No camera connected"
        lmain = tk.Label(CVFrame1, text="No camera connected.. \n Connect a camera to start computer vision",
                         font=('Century Gothic', 14), bg="#D0D3D4")
        lmain.place(relx=0.5, rely=0.5, anchor="center")
    else:
        # Create a label in the frame for displaying the camera feed
        lmain = tk.Label(CVFrame1, highlightthickness=0, bd=0)
        lmain.place(x=-10, y=55)

        # Create a Tkinter compatible image
        tk_image = None

        # function for video streaming
        def video_stream():
            nonlocal tk_image
            while True:
                _, frame = cap.read()
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                img = Image.fromarray(cv2image)

                # update the tk_image
                tk_image = ImageTk.PhotoImage(image=img)

                # wait for 10 milliseconds before capturing the next frame
                time.sleep(0.01)

        # function for updating the label with the tk_image
        def update_label():
            nonlocal tk_image
            if tk_image is not None:
                lmain.configure(image=tk_image)
                lmain.image = tk_image  # prevent garbage collection
            # schedule the update to happen again after 10ms
            lmain.after(10, update_label)

        # start the video streaming in a separate thread
        t = threading.Thread(target=video_stream)
        t.daemon = True
        t.start()

        # start the label update loop in the main thread
        update_label()

########################################################################################
def process_input():
    while assistant_running:
        query = take_command().lower()
        user_label.config(text=query.capitalize())
        AssistantFrame.update()
        if 'hello' in query:
            ai_response = "Hello! Please tell me how can I help you."
            response_label.config(text=textwrap.fill(ai_response, width=100))
            AssistantFrame.update()
            speak(ai_response)
            while True:
                query = take_command().lower()
                user_label.config(text=query.capitalize())
                AssistantFrame.update()
                if 'search' in query:
                    ai_response = "What do you want me to search for?"
                    response_label.config(text=textwrap.fill(ai_response, width=100))
                    AssistantFrame.update()
                    speak(ai_response)
                    query = take_command().lower()
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
                    exit(0)
                    response_label.config(text="")
                    user_label.config(text="")
                    AssistantFrame.update()
                else:
                    response_label.config(text="Sorry, I didn't understand.\n Please try again or say 'exit' to quit.")
                    AssistantFrame.update()
                    ai_response= "Sorry, I didn't understand. Please try again or say 'exit' to quit. "
                    speak(ai_response)
            # if user input is not "hello" or "exit/goodbye"
            else:
                response_label.config(
                    text="Sorry, I didn't understand.\nPlease start the voice assistant again.")
                speak("Sorry, I didn't understand. Please start the voice assistant again.")
        time.sleep(1)

def start_assistant():
    global assistant_running
    assistant_running = True
    response_label.config(text="Hello! Nice to meet you!")
    response_label.update()
    ai_response= "Hello! Nice to meet you!"
    t = threading.Thread(target=speak, args=(ai_response,))
    t.start()
    t = threading.Thread(target=process_input)
    t.start()

def stop_assistant():
    global assistant_running
    assistant_running = False

########################################################################################

def Game1():
    t = threading.Thread(target=Game)
    t.start()

def Game():
    # display main menu
    response = ("Welcome to Get to the University Game!")
    strlabel.config(text=response)
    strlabel.update()
    speak(".....Welcome to Get to the University Game!")
    strlabel.config(text="The game where you try to get in on time to your lessons!")
    strlabel.update()
    speak("The game where you try to get in on time to your lessons!")
    strlabel.config(text="Would you like to play?")
    strlabel.update()
    speak("Would you like to play?")

    def get_user_input():
        while True:
            play = take_command().lower()
            user_label1.config(text=play)

            # Handle user input
            if play == "yes":
                strlabel.config(text="Let's go!")
                strlabel.update()
                speak("Let's go!")
                time.sleep(1)
                t = threading.Thread(target=gamerun())
                t.start()
                strlabel.config(text="Do you want to play again?")
                strlabel.update()
                speak("Do you want to play again?")
                break
            elif play == "no":
                strlabel.config(text="Bye bye!")
                strlabel.update()
                speak("Bye bye!")
                stop_function()
                return
            elif play == "exit":
                strlabel.config(text="Bye bye!")
                strlabel.update()
                speak("Bye bye!")
                stop_function()
                return
            else:
                strlabel.config(text="I didn't catch that. Please say 'yes' or 'no'.")
                strlabel.update()
                speak("I didn't catch that. Please say 'yes' or 'no'.")

        while True:
            again = take_command().lower()
            if again == "yes":
                strlabel.config(text="Let's go again!")
                strlabel.update()
                speak("Let's go again!")
                t = threading.Thread(target=gamerun())
                t.start()
                strlabel.config(text="Do you want to play again?")
                strlabel.update()
                speak("Do you want to play again?")
            elif again == "no":
                strlabel.config(text="Thanks for playing!")
                strlabel.update()
                speak("Thanks for playing!")
                break
            else:
                strlabel.config(text="I didn't catch that. Please say 'yes' or 'no'.")
                strlabel.update()
                speak("I didn't catch that. Please say 'yes' or 'no'.")

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

        strlabel.config(text="say continue to start game or exit to stop")
        strlabel.update()
        speak("say continue to start game or exit to stop")
        response = take_command().lower()
        if response == "exit":
            speak("Exiting game.")
            break
        elif response != "continue":
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
                #random Choice for selecting the path
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
                    strlabel.config(text="say continue to proceed")
                    strlabel.update()
                    speak("say continue to proceed")
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
    user_label1.config(text = " ")
    return
    show_frame(MainPage)

def vinput():
    # validates inputs so it can't crash and the choice can only be a, b or c
    valid_inputs = {"number one", "number 2", "number 3"}
    while True:
        try:
            inp = take_command().lower()
        except:
            # not strictly needed but in case something could happen
            # this is there as a fall back just in case, slightly unnecessary
            speak("That's not anything!")
            continue
        if inp in valid_inputs:
            break
        else:
            speak("That's not a valid input")
    return inp

def Money():
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
    speak(f"It took you {timetaken} minutes to handle the situation.")
    return timetaken

def river():
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
        strlabel.config(text="You Managed to steal the boat and cross the river in time \n but the police will be looking for you! ")
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
    speak(f"It took you {timetaken} minutes to handle the situation.")
    return timetaken

def AppleStore():
    strlabel.config(text="As you approach the AppleStore, you hear loud noises and see a commotion.")
    strlabel.update()
    speak("As you approach the AppleStore, you hear loud noises and see a commotion.")
    strlabel.config(text="You notice people breaking into the store and taking iPads, iPhones, and Macbooks.")
    strlabel.update()
    speak("You notice people breaking into the store and taking iPads, iPhones, and Macbooks.")
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
    timetaken=0
    strlabel.config(text="You are on your way and come across a drugged grizzly bear \n which escaped from the local zoo.")
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

def ComputervisionPage1():
    show_frame(ComputerVisionPage)
    response = " Computer Vision Page Selected "
    t = threading.Thread(target=speak1, args=(response,))
    t.start()
    Camera()

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

def Back_btn1():
    show_frame(MainPage)
    response = " Back Button Selected "
    t = threading.Thread(target=speak1, args=(response,))
    t.start()
    stop_function()

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


clock_label = tk.Label(MainPage, fg="white", bg="#004aad", height=1, font=('Century Gothic', 25))
clock_label.pack()
clock_label.place(x=1000, y=80)

update_time()

#################################--------------Computer Vision PAGE------------------############################################################
ComputerVisionPage = tk.Frame(window, width=1366, height=768)
ComputerVisionPage.grid(row=0, column=0, stick='nsew')
background_label = tk.Label(ComputerVisionPage, image=filename1)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

CVFrame = tk.Frame(ComputerVisionPage, bg="white")
CVFrame.place(relx=0.2, rely=0.32, relwidth=0.25, relheight=0.40)

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

clock = tk.Label(ComputerVisionPage, fg="white", bg="#004aad", height=1, font=('Century Gothic', 25))
clock.pack()
clock.place(x=1000, y=80)
tick()

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
