import cv2
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.uix.button import Button
from kivy.uix.label import Label
from PIL import Image as PILImage

Window.size = (380, 650)
Window.clearcolor = (1, 1, 1, 1)

# import camera helpers from appfunction
from appfunction import start_camera, stop_camera, camera_update, save_uploaded_image, CAMERA_ORIENTATION

# Main screen where user chooses language
class MainWindow(Screen):
    pass

# Filipino language window
class FilipinoWindow(Screen):
    pass

# English language window
class EnglishWindow(Screen):
    pass

# HandSigns screen (for camera feed)
class HandSigns(Screen):
    def on_enter(self):
        start_camera(self, src=0, interval=1.0/30.0)

    def update(self, dt):
        # kept for compatibility; delegate to shared camera_update
        camera_update(self, dt)

    def on_leave(self):
        stop_camera(self)

    def upload_photo(self):
        filechooser = FileChooserIconView()
        filechooser.bind(on_selection=self.on_file_select)
        popup = Popup(title="Select a Photo", content=filechooser, size_hint=(0.9, 0.9))
        popup.open()

    def on_file_select(self, instance, value):
        if value:
            img_path = value[0]
            ok = save_uploaded_image(img_path)
            if ok:
                print(f"Image saved as: {img_path}")
            else:
                print("Failed to save image")

# HandSign screen (SECOND CAMERA)
class HandSign(Screen):
    def on_enter(self):
        start_camera(self, src=0, interval=1.0/30.0)

    def update(self, dt):
        camera_update(self, dt)

    def on_leave(self):
        stop_camera(self)

    def upload_photo(self):
        filechooser = FileChooserIconView()
        filechooser.bind(on_selection=self.on_file_select)
        popup = Popup(title="Select a Photo", content=filechooser, size_hint=(0.9, 0.9))
        popup.open()

    def on_file_select(self, instance, value):
        if value:
            img_path = value[0]
            ok = save_uploaded_image(img_path)
            if ok:
                print(f"Image saved as: {img_path}")
            else:
                print("Failed to save image")

# Instructions window
class InstructionsWindow(Screen):
    def on_enter(self):
        if self.manager.get_screen('filipino'):
            self.ids.instructions_label.text = (
                "Filipino Sign Language Instructions:\n\n"
                "1. Gesture for 'Thank You' is made by placing your hand near your chin, then moving it away.\n"
                "2. Gesture for 'Please' is made by touching your chest with an open hand and moving it outward."
            )
        elif self.manager.get_screen('english'):
            self.ids.instructions_label.text = (
                "English Sign Language Instructions:\n\n"
                "1. 'A' is formed by making a fist with your thumb on the outside.\n"
                "2. 'B' is made by forming a flat hand with fingers together."
            )

class InstructionsWindow1(Screen):
    def on_enter(self):
        if self.manager.get_screen('filipino'):
            self.ids.instructions_label.text = (
                "Filipino Sign Language Instructions:\n\n"
                "1. Gesture for 'Thank You' is made by placing your hand near your chin, then moving it away.\n"
                "2. Gesture for 'Please' is made by touching your chest with an open hand and moving it outward."
            )
        elif self.manager.get_screen('english'):
            self.ids.instructions_label.text = (
                "English Sign Language Instructions:\n\n"
                "1. 'A' is formed by making a fist with your thumb on the outside.\n"
                "2. 'B' is made by forming a flat hand with fingers together."
            )

class WindowManager(ScreenManager):
    pass

kv = Builder.load_file("my.kv")

class MyApp(App):
    def build(self):
        return kv
    
if __name__ == "__main__":
    MyApp().run()
