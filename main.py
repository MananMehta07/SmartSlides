from cvzone.HandTrackingModule import HandDetector
import cv2
import os
import numpy as np
import json
import speech_recognition as sr
import pyttsx3
import threading

# Parameters
width, height = 1280, 720  # Presentation display dimensions
folderPath = "Presentation"

# Camera Setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Hand Detector
detectorHand = HandDetector(detectionCon=0.8, maxHands=1)

# Speech Recognition and Text-to-Speech Setup
recognizer = sr.Recognizer()
engine = pyttsx3.init()
engine.setProperty('rate', 125)  # Slow down the voice rate

# Variables
delay = 30
buttonPressed = False
counter = 0
imgNumber = 0
delayCounter = 0
annotations = [[]]
annotationNumber = -1
annotationStart = False
hs, ws = 240, 320  # Embedded camera feed dimensions
displayText = ""  # Text to display the detected functionality
textCounter = 0  # Counter for clearing text display
annotationColor = (255, 0, 255)  # Fixed annotation color
cursorColor = (255, 255, 255)  # Cursor color for other actions

# Smoothing parameters
smoothFactor = 5  # Higher value = smoother movements
prevX, prevY = 0, 0

# Get list of presentation images
pathImages = sorted(os.listdir(folderPath), key=len)
print(pathImages)

# Voice Command Function
def execute_voice_command(command):
    global imgNumber, annotations, annotationNumber, displayText

    if "next slide" in command:
        if imgNumber < len(pathImages) - 1:
            imgNumber += 1
            annotations = [[]]
            annotationNumber = -1
            displayText = "Next Slide"
            engine.say("Moving to the next slide.")
            engine.runAndWait()

    elif "previous slide" in command:
        if imgNumber > 0:
            imgNumber -= 1
            annotations = [[]]
            annotationNumber = -1
            displayText = "Previous Slide"
            engine.say("Moving to the previous slide.")
            engine.runAndWait()

    elif "clear annotations" in command:
        annotations = [[]]
        annotationNumber = -1
        displayText = "Annotations Cleared"
        engine.say("All annotations cleared.")
        engine.runAndWait()

    elif "go to slide" in command:
        try:
            slide_number = int(command.split("slide")[-1].strip())
            if 0 < slide_number <= len(pathImages):
                imgNumber = slide_number - 1
                annotations = [[]]
                annotationNumber = -1
                displayText = f"Slide {slide_number}"
                engine.say(f"Going to slide {slide_number}.")
                engine.runAndWait()
        except ValueError:
            engine.say("Invalid slide number.")
            engine.runAndWait()


# Background Listener Function
def listen_for_voice_commands():
    while True:
        try:
            with sr.Microphone() as source:
                print("Listening for voice commands...")
                audio = recognizer.listen(source, timeout=2, phrase_time_limit=5)
                command = recognizer.recognize_google(audio).lower()
                print("Voice command:", command)
                execute_voice_command(command)
        except sr.UnknownValueError:
            print("Could not understand the command.")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
        except sr.WaitTimeoutError:
            pass

# Start a thread for voice commands
voice_thread = threading.Thread(target=listen_for_voice_commands, daemon=True)
voice_thread.start()

while True:
    # Get image frame
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip the camera feed horizontally
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgSlide = cv2.imread(pathFullImage)

    # Resize slide to fit the screen while maintaining aspect ratio
    slideAspect = imgSlide.shape[1] / imgSlide.shape[0]  # width/height of the slide
    if slideAspect > width / height:  # Slide is wider than the screen
        newWidth = width
        newHeight = int(width / slideAspect)
    else:  # Slide is taller than the screen
        newHeight = height
        newWidth = int(height * slideAspect)
    imgCurrent = cv2.resize(imgSlide, (newWidth, newHeight))

    # Replace camera feed background with the slide
    mask = np.zeros((height, width, 3), dtype=np.uint8)
    yOffset = (height - newHeight) // 2
    xOffset = (width - newWidth) // 2
    mask[yOffset:yOffset + newHeight, xOffset:xOffset + newWidth] = imgCurrent
    imgCurrent = mask

    # Find the hand and its landmarks
    hands, img = detectorHand.findHands(img)  # with draw

    if hands and buttonPressed is False:  # If hand is detected

        hand = hands[0]
        cx, cy = hand["center"]
        lmList = hand["lmList"]  # List of 21 Landmark points
        fingers = detectorHand.fingersUp(hand)  # List of which fingers are up

        # Gesture controls for slides
        if fingers == [1, 0, 0, 0, 0]:  # Right swipe
            displayText = "Previous Slide"
            cursorColor = (0, 255, 0)
            print(displayText)
            buttonPressed = True
            if imgNumber > 0:
                imgNumber -= 1
                annotations = [[]]
                annotationNumber = -1

        if fingers == [0, 0, 0, 0, 1]:  # Left swipe
            displayText = "Next Slide"
            cursorColor = (0, 0, 255)
            print(displayText)
            buttonPressed = True
            if imgNumber < len(pathImages) - 1:
                imgNumber += 1
                annotations = [[]]
                annotationNumber = -1

        # Highlighting
        if fingers == [0, 1, 1, 0, 0]:  # Highlight point
            displayText = "Highlighting"
            cursorColor = (255, 255, 0)
            print(displayText)
            cv2.circle(imgCurrent, (cx, cy), 12, cursorColor, cv2.FILLED)

        # Drawing annotations
        if fingers == [0, 1, 0, 0, 0]:  # Draw
            displayText = "Drawing"
            print(displayText)
            cursorColor = (255, 0, 255)  # Fixed color for annotations
            if annotationStart is False:
                annotationStart = True
                annotationNumber += 1
                annotations.append([])
            annotations[annotationNumber].append((cx, cy))
            cv2.circle(imgCurrent, (cx, cy), 12, annotationColor, cv2.FILLED)

        else:
            annotationStart = False

        # Undo annotations
        if fingers == [0, 1, 1, 1, 0]:  # Undo
            displayText = "Undo"
            cursorColor = (0, 255, 255)
            print(displayText)
            if annotations:
                annotations.pop(-1)
                annotationNumber -= 1
                buttonPressed = True

        # Erase all annotations
        if fingers == [1, 1, 0, 0, 0]:  # Closed fist
            displayText = "Erase All Annotations"
            cursorColor = (255, 0, 0)
            print(displayText)
            annotations = [[]]
            annotationNumber = -1

    if buttonPressed:
        counter += 1
        if counter > delay:
            counter = 0
            buttonPressed = False

    for i, annotation in enumerate(annotations):
        for j in range(len(annotation)):
            if j != 0:
                cv2.line(imgCurrent, annotation[j - 1], annotation[j], annotationColor, 12)

    # Resize the live camera feed
    imgSmall = cv2.resize(img, (ws, hs))  # Resize to larger dimensions

    # Display the functionality text on the embedded camera feed
    if displayText:
        cv2.putText(imgSmall, displayText, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, cursorColor, 2)

    imgCurrent[0:hs, width - ws:width] = imgSmall  # Place the resized feed at the top-right corner

    # Display the slides with the embedded camera feed in full-screen mode
    cv2.namedWindow("Slides", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Slides", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Slides", imgCurrent)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
