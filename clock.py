import customtkinter
import cv2
from PIL import Image, ImageTk
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components import containers
import time
from datetime import datetime
import urllib.request
import os

# ── Download model if not present ────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        MODEL_PATH
    )
    print("Model downloaded!")

# Constants
FPS_DELAY    = 10  # 10ms = 100fps, 16ms = 60fps
CLOCK_UPDATE = 1000 
GESTURE_HOLD = 0.01

# ── MediaPipe New API Setup
latest_result = None

def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

base_options   = python.BaseOptions(model_asset_path=MODEL_PATH)
hand_options   = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=20,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
    result_callback=result_callback
)
detector = vision.HandLandmarker.create_from_options(hand_options)

# ── App Window
customtkinter.set_appearance_mode("dark")

root = customtkinter.CTk()
root.geometry("550x500")
root.title("Handy Clock")

# ── Clock Overlay Window 
clock_win = customtkinter.CTkToplevel(root)
clock_win.title("Clock")
clock_win.geometry("400x200")
clock_win.attributes("-topmost", True)
clock_win.attributes("-alpha", 0.85)
clock_win.overrideredirect(True)
clock_win.withdraw()

screen_w = root.winfo_screenwidth()
screen_h = root.winfo_screenheight()
clock_win.geometry(f"400x200+{screen_w//2 - 50}+{screen_h//2}")
#clock_win.geometry(f"400x200+{screen_w//2 - 100}+{screen_h//2 - 50}")

clock_label = customtkinter.CTkLabel(
    clock_win, text="00:00:00",
    font=("Helvetica", 64, "bold"),
    text_color="#a8a8a8"
)
clock_label.pack(expand=True, fill="both", padx=20, pady=20)

date_label = customtkinter.CTkLabel(
    clock_win, text="",
    font=("Helvetica", 30),
    text_color="#aaaaaa"
)
date_label.pack(pady=(0, 10))

def update_clock():
    now = datetime.now()
    clock_label.configure(text=now.strftime("%H:%M:%S"))
    date_label.configure(text=now.strftime("%A, %d %B %Y"))
    root.after(CLOCK_UPDATE, update_clock)

update_clock()

# ── Main UI 
main_frame = customtkinter.CTkFrame(root)
main_frame.pack(fill="both", expand=True, padx=10, pady=10)

cam_frame = customtkinter.CTkFrame(main_frame)
cam_frame.pack(side="top", padx=10, pady=10)

cam_label = customtkinter.CTkLabel(cam_frame, text="")
cam_label.pack()

btn_frame = customtkinter.CTkFrame(root)
btn_frame.pack(pady=8)

cap = None
clock_visible   = False
gesture_start_time = None
frame_timestamp = 0

# ── Gesture Detection 
def is_pointing_up(landmarks):
    """Index tip above knuckle, other fingers curled."""
    lm = landmarks
    index_up    = lm[8].y  < lm[6].y
    middle_down = lm[12].y > lm[10].y
    ring_down   = lm[16].y > lm[14].y
    pinky_down  = lm[20].y > lm[18].y
    return index_up and middle_down and ring_down and pinky_down

def show_clock():
    global clock_visible
    clock_visible = True
    clock_win.deiconify()
    clock_win.lift()

def hide_clock():
    global clock_visible
    clock_visible = False
    clock_win.withdraw()

# ── Camera Loop 
def update_frame():
    global cap, gesture_start_time, frame_timestamp

    if cap and cap.isOpened():
        ret, frame_img = cap.read()
        if ret:
            frame_img = cv2.flip(frame_img, 1)
            rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)

            # Send to MediaPipe
            frame_timestamp += 1
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            detector.detect_async(mp_image, frame_timestamp)

            pointing = False

            # Draw results from latest callback
            if latest_result and latest_result.hand_landmarks:
                for hand_landmarks in latest_result.hand_landmarks:
                    # Draw landmarks manually
                    h, w, _ = frame_img.shape
                    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

                    # Draw connections - hardcoded cause version conflicts
                    connections = [
                        (0,1),(1,2),(2,3),(3,4),         # thumb
                        (0,5),(5,6),(6,7),(7,8),         # index
                        (0,9),(9,10),(10,11),(11,12),    # middle
                        (0,13),(13,14),(14,15),(15,16),  # ring
                        (0,17),(17,18),(18,19),(19,20),  # pinky
                        (5,9),(9,13),(13,17)             # palm
                    ]
                    for start, end in connections:
                        cv2.line(frame_img, points[start], points[end], (0, 200, 150), 2)
                    for pt in points:
                        cv2.circle(frame_img, pt, 4, (0, 255, 180), -1)

                    if is_pointing_up(hand_landmarks):
                        pointing = True
                        cv2.putText(frame_img, "POINTING UP", (10, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 180), 2)

            # Gesture hold logic

            if pointing:
                if gesture_start_time is None:
                    gesture_start_time = time.time()
                elif time.time() - gesture_start_time >= GESTURE_HOLD:
                    if not clock_visible:
                        show_clock()
            else:
                gesture_start_time = None
                if clock_visible:
                    hide_clock()

            img = Image.fromarray(cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB))
            img = img.resize((600, 450))
            ctk_img = ImageTk.PhotoImage(img)
            cam_label.configure(image=ctk_img, text="")
            cam_label.image = ctk_img

    root.after(FPS_DELAY, update_frame)

def start_camera():
    global cap
    if cap is not None:
        return  # Camera already started
    cap = cv2.VideoCapture(0)
    update_frame()

def stop_camera():
    global cap
    if cap:
        cap.release()
        cap = None
    cam_label.configure(image="", text="Camera stopped")
    hide_clock()

customtkinter.CTkButton(btn_frame, text="▶  Start Camera", command=start_camera).pack(side="left", padx=10)
customtkinter.CTkButton(btn_frame, text="■  Stop Camera",  command=stop_camera).pack(side="left", padx=10)

root.mainloop()