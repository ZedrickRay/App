# File: test.py

import cv2
import pyttsx3
import numpy as np
import time
import tensorflow as tf
from multiprocessing import Process
from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from PIL import Image as PILImage
from cvzone.HandTrackingModule import HandDetector
import os

# Load labels
with open("Model/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize TFLite interpreter
interpreter = tf.lite.Interpreter(model_path="Model/model_unquant.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

Window.size = (380, 650)
Window.clearcolor = (1, 1, 1, 1)

def speak_in_process(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def update_ui_label(dt, widget, text):
    if widget:
        widget.text = text

def predict_image(img):
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.squeeze(output)
    index = int(np.argmax(prediction))
    return prediction, index

class MainWindow(Screen): pass
class FilipinoWindow(Screen): pass
class EnglishWindow(Screen): pass

class SignBase(Screen):
    def on_enter(self):
        self.capture = cv2.VideoCapture(0)
        self.detector = HandDetector(maxHands=1)
        self.predicted_text = ""
        self.last_label = ""
        self.last_spoken_time = 0
        self.event = Clock.schedule_interval(self.update, 1.0 / 30.0)

    def speak(self, text):
        Process(target=speak_in_process, args=(text,), daemon=True).start()

    def speak_full_text(self):
        editable = self.ids.get("editable_text")
        if editable:
            text = editable.text.strip()
            if text:
                self.speak(text)

    def upload_photo(self):
        try:
            filechooser = FileChooserIconView()
            filechooser.filters = ['*.png', '*.jpg', '*.jpeg']
            filechooser.bind(on_selection=self.on_file_select)
            popup = Popup(title="Select a Photo", content=filechooser, size_hint=(0.9, 0.9))
            self._popup = popup
            popup.open()
        except Exception as e:
            print("Error opening file chooser:", e)

    def on_file_select(self, instance, selection):
        if selection:
            try:
                selected_file = selection[0]
                img = PILImage.open(selected_file).convert("RGB")
                img_resized = img.resize((224, 224))
                img_np = np.array(img_resized)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                _, index = predict_image(img_bgr)
                label = labels[index] if 0 <= index < len(labels) else "?"

                self.predicted_text += label
                Clock.schedule_once(lambda dt: self.update_editable_text())
                Clock.schedule_once(lambda dt: update_ui_label(dt, self.ids.get("text_output"), f"Detected: {label}"))
                self.speak(label)

                if hasattr(self, '_popup'):
                    self._popup.dismiss()
                print(f"Image prediction: {label}")
            except Exception as e:
                print("Error predicting uploaded image:", e)

    def update(self, dt):
        try:
            ret, frame = self.capture.read()
            if not ret:
                return

            frame = cv2.flip(frame, 1)
            hands, _ = self.detector.findHands(frame, draw=False)

            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                offset = 20
                x1 = max(0, x - offset)
                y1 = max(0, y - offset)
                x2 = min(frame.shape[1], x + w + offset)
                y2 = min(frame.shape[0], y + h + offset)
                hand_img = frame[y1:y2, x1:x2]

                if hand_img.size > 0:
                    _, index = predict_image(hand_img)
                    label = labels[index] if 0 <= index < len(labels) else "?"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 0, 0), 2, cv2.LINE_AA)

                    now = time.time()
                    if label != self.last_label and (now - self.last_spoken_time) > 1:
                        self.last_label = label
                        self.last_spoken_time = now
                        self.predicted_text += label

                        Clock.schedule_once(lambda dt: self.update_editable_text())
                        self.speak(label)

                    Clock.schedule_once(lambda dt: update_ui_label(
                        dt, self.ids.get("text_output"), f"Detected: {label}"))

            frame = cv2.flip(frame, 0)
            buf = frame.tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            if self.ids.get("cam_view"):
                self.ids.cam_view.texture = texture

        except Exception as e:
            print("Update error:", e)

    def update_editable_text(self):
        if self.ids.get("editable_text"):
            self.ids.editable_text.text = f"Sentence: {self.predicted_text}"

    def clear_text(self):
        self.predicted_text = ""
        self.last_label = ""
        self.last_spoken_time = 0
        if self.ids.get("text_output"):
            self.ids.text_output.text = "Detected: "
        if self.ids.get("sentence_output"):
            self.ids.sentence_output.text = "Sentence: "
        if self.ids.get("editable_text"):
            self.ids.editable_text.text = "Sentence: "

    def on_leave(self):
        if hasattr(self, 'event'):
            self.event.cancel()
        if hasattr(self, 'capture'):
            self.capture.release()

class HandSigns(SignBase): pass
class HandSign(SignBase): pass
class InstructionsWindow(Screen): pass
class InstructionsWindow1(Screen): pass
class WindowManager(ScreenManager): pass

kv = Builder.load_file("my.kv")

class MyApp(App):
    def build(self):
        return kv

if __name__ == "__main__":
    MyApp().run()
