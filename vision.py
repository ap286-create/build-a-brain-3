import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles
from ultralytics import YOLO
from ultralytics.engine.results import Results


"""
GOAL: hand/face/object detection wrappers; mediapipe, drawing, callbacks
"""
class VisionSystem:
    def __init__(self):
        self.face_result = None
        self.hand_result = None
        self.yolo = YOLO("models/yolov8n.pt")
        self.yolo_result = None

        self.timestamp = 0
        self.command = None

        self.gesture_start_time = None
        self.last_trigger_time = 0
        self.cooldown_time = 0.3
        self.hold_time = 0.3
        self.active_gesture = None

        self.expression_label = None
        self.expression_scores = {}

        self.pointer_x = None
        self.pointer_y = None

        self.active_object = None
        self.last_yolo_time = 0
        self.yolo_interval = 0.1
        self.object_metrics = None

        #main set up landmarkers/object dectectors (had to move here for print_face_result to work)
        self.face_landmarker = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=face_model_path),
                running_mode=VisionRunningMode.LIVE_STREAM,
                result_callback=self.print_face_result,
                output_face_blendshapes=True
            ))
        self.hand_landmarker = HandLandmarker.create_from_options(HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=hand_model_path),
                running_mode=VisionRunningMode.LIVE_STREAM,
                result_callback=self.print_hand_result,
                num_hands=2
            ))
        
    #(creating landmarker instances in live stream mode)
    def print_face_result(self, result, output_image, timestamp_ms):
        self.face_result = result.face_landmarks
        if result.face_blendshapes:
            label, scores = self.classify_expression(result.face_blendshapes[0])
            self.expression_label = label
            self.expression_scores = scores
        else:
            self.expression_label = None

    # handles hand logic such as thumbs up and thumbs down. 
    def print_hand_result(self, result, output_image, timestamp_ms):
        self.hand_result = result.hand_landmarks
        if not self.hand_result:
            self.pointer_x = None
            self.pointer_y = None
            self.gesture_start_time = None
            self.active_gesture = None
            return
        hand = self.hand_result[0]   # first detected hand
        tip = hand[8]                # index finger tip

        self.pointer_x = tip.x
        self.pointer_y = tip.y
        
        #thumbs up/down part (make sure intentional gesture + doesn't move on from slide to soon)
        now = time.time()
        
        if now - self.last_trigger_time < self.cooldown_time:
           # self.gesture_start_time = None
            #self.active_gesture = None   # 🔥 ADD THIS
            return

        if not self.hand_result:
            self.pointer_x = None
            self.pointer_y = None
            self.gesture_start_time = None
            self.active_gesture = None   # 🔥 ADD THIS
            return

        detected = None

        #which gesture?
        for hand in self.hand_result:
            up, down = gesture_confidence(hand)
            if down > 0.6:
                detected = "skip_two"
                break
            if up > 0.6:
                detected = "next_slide"
                break

        if detected is None:
            self.gesture_start_time = None
            self.active_gesture = None
            return
        
        if self.gesture_start_time is None:
            self.active_gesture = None
            
        # HOLD LOGIC (shared, applies to both gestures)
        if detected != self.active_gesture:
            self.active_gesture = detected
            self.gesture_start_time = now

        if now - self.gesture_start_time >= self.hold_time:
            self.command = detected
            self.last_trigger_time = now
            self.active_gesture = None
            self.gesture_start_time = None 

    def prepare(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return mp_image, frame
    
    def process(self, mp_image, frame, state):
        self.timestamp += 1
        ts = self.timestamp
        if state.models["face"]:
            self.face_landmarker.detect_async(mp_image, ts)
        else:
            self.face_result = None
            self.expression_label = None
        
        self.hand_landmarker.detect_async(mp_image, ts)

        if state.models["object"]:
            now = time.time() #so doesn't over like caluclate and go bonkers
            if now - self.last_yolo_time > self.yolo_interval:
                results = self.yolo(frame, verbose=False)
                self.yolo_result = results[0]
                self.last_yolo_time = now
        else:
            self.yolo_result = None

    def render(self, frame):
        if self.face_result:
            frame = draw_landmarks_on_image(frame, self.face_result)
        if self.hand_result:
            frame = draw_hand_landmarks(frame, self.hand_result)
        if self.yolo_result:
            frame = self.yolo_result.plot()

        if self.expression_label:
            cv2.putText(frame, self.expression_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (141,68,68), 2)
        return frame

    def get_command(self):
        cmd = self.command
        self.command = None
        return cmd
    
    def classify_expression(self, blendshapes):
        scores = {
            "happy": 0,
            "surprised": 0,
            "sad": 0,
            "confused": 0,
            "neutral": 0
        }

        for b in blendshapes:
            name = b.category_name
            score = b.score

            # -------- HAPPY (stronger separation) --------
            if name in ["mouthSmileLeft", "mouthSmileRight"]:
                scores["happy"] += score * 1.8
            if name in ["cheekSquintLeft", "cheekSquintRight"]:
                scores["happy"] += score * 1.2

            # -------- SURPRISED --------
            if name == "jawOpen":
                scores["surprised"] += score * 1.6
            if name in ["eyeWideLeft", "eyeWideRight"]:
                scores["surprised"] += score * 1.4

            # -------- SAD --------
            if name in ["mouthFrownLeft", "mouthFrownRight"]:
                scores["sad"] += score * 1.7
            if name == "browInnerUp":
                scores["sad"] += score * 0.8

            # -------- CONFUSED --------
            if name in ["browDownLeft", "browDownRight"]:
                scores["confused"] += score * 1.6
            if name == "browInnerUp":
                scores["confused"] += score * 1.0

        # -------- stronger dominance rule --------
        best = max(scores, key=scores.get)
        best_score = scores[best]

        # second highest check (important for stability)
        sorted_scores = sorted(scores.values(), reverse=True)
        second_best = sorted_scores[1] if len(sorted_scores) > 1 else 0

        # require clear separation
        if best_score < 0.4 or (best_score - second_best < 0.15):
            return "neutral", scores

        return best, scores
    #######
    def get_boxes(self):
        if self.yolo_result is None:
            return []
        boxes = []

        for box in self.yolo_result.boxes:
            cls_id = int(box.cls[0])
            label = self.yolo_result.names[cls_id]

            if label == "person":
                continue 

            xyxy = box.xyxy[0].cpu().numpy()
            boxes.append((xyxy,label))

        return boxes
    #######
    
                

#actually drawing the landmarks (taken straight frm the gitbhub for face, chat for hand)
def draw_landmarks_on_image(rgb_image, face_landmarks_list):
    annotated_image = np.copy(rgb_image)
    subtle_style = drawing_utils.DrawingSpec(color=(180,180,180), thickness=1, circle_radius=1)

    for face_landmarks in face_landmarks_list:
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            landmark_drawing_spec=subtle_style,
            connection_drawing_spec=subtle_style)
    return annotated_image

#hand time
def draw_hand_landmarks(image, hand_landmarks):
    for hand in hand_landmarks:
        for lm in hand:
            h, w, _ = image.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 5, (0,255,0), -1)
    return image
#### we needa change this to return on screen what i want not just yolo shtuff
def draw_object_detections():
    pass
#####

# setting up stuff for face_landmarker
face_model_path = "models/face_landmarker.task"
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions

# setting up stuff for hand_landmarker
hand_model_path = "models/hand_landmarker.task"
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions


BaseOptions = python.BaseOptions
VisionRunningMode = vision.RunningMode


#functions to detect camera stuff like thumbsup, pointing, thumbsdown, etc
# def is_thumbs_up(hand_landmarks):
#     # landmark indices (MediaPipe standard)
#     THUMB_TIP = 4
#     THUMB_IP = 3

#     INDEX_TIP = 8
#     MIDDLE_TIP = 12
#     RING_TIP = 16
#     PINKY_TIP = 20

#     # y is inverted (top = smaller value)
#     thumb_up = hand_landmarks[THUMB_TIP].y < hand_landmarks[THUMB_IP].y

#     fingers_down = (
#         hand_landmarks[INDEX_TIP].y > hand_landmarks[THUMB_IP].y and
#         hand_landmarks[MIDDLE_TIP].y > hand_landmarks[THUMB_IP].y and
#         hand_landmarks[RING_TIP].y > hand_landmarks[THUMB_IP].y and
#         hand_landmarks[PINKY_TIP].y > hand_landmarks[THUMB_IP].y
#     )
#     return thumb_up and fingers_down

# def is_thumbs_down(hand_landmarks):
#     THUMB_TIP = 4
#     THUMB_IP = 3

#     INDEX_TIP = 8
#     MIDDLE_TIP = 12
#     RING_TIP = 16
#     PINKY_TIP = 20

#     thumb_down = hand_landmarks[THUMB_TIP].y > hand_landmarks[THUMB_IP].y

#     fingers_down = (
#         hand_landmarks[INDEX_TIP].y > hand_landmarks[THUMB_IP].y and
#         hand_landmarks[MIDDLE_TIP].y > hand_landmarks[THUMB_IP].y and
#         hand_landmarks[RING_TIP].y > hand_landmarks[THUMB_IP].y and
#         hand_landmarks[PINKY_TIP].y > hand_landmarks[THUMB_IP].y
#     )

#     return thumb_down and fingers_down

#more helper funcitons (gesture confidence, object filtering)
def gesture_confidence(hand):
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20

    wrist_y = hand[WRIST].y

    # ---------- THUMB DIRECTION ----------
    thumb_up_score = max(0, wrist_y - hand[THUMB_TIP].y)
    thumb_down_score = max(0, hand[THUMB_TIP].y - wrist_y)

    # normalize (prevents scale issues)
    thumb_up_score = min(1.0, thumb_up_score * 5)
    thumb_down_score = min(1.0, thumb_down_score * 5)

    # ---------- FINGER STATE ----------
    fingers_folded = 0
    for tip in [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]:
        if hand[tip].y > wrist_y:
            fingers_folded += 1

    finger_score = fingers_folded / 4  # 0 → 1

    # ---------- FINAL SCORES ----------
    thumbs_up_conf = (0.6 * thumb_up_score) + (0.4 * finger_score)
    thumbs_down_conf = (0.6 * thumb_down_score) + (0.4 * finger_score)

    return thumbs_up_conf, thumbs_down_conf

#ol activity
def get_object_color_hex(frame, box):
    x1, y1, x2, y2 = map(int, box)

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    size = 10  # small square
    crop = frame[
        max(0, cy-size):cy+size,
        max(0, cx-size):cx+size
    ]

    if crop.size == 0:
        return None, (0,0,0)

    avg = crop.mean(axis=(0,1))
    b, g, r = avg

    hex_color = "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))
    return hex_color, (int(b), int(g), int(r))

def draw_color_indicator(frame, box, bgr):
    x1, y1, x2, y2 = map(int, box)
    cx = (x1 + x2) // 2
    cy = y1 - 10

    cv2.circle(frame, (cx, cy), 6, bgr, -1)
    return frame

def redness_score(bgr):
    b, g, r = bgr
    return r - max(g, b)  # how dominant red is

#pl activity
def pick_active_object(frame, boxes):
    if not boxes:
        return None
    
    h, w = frame.shape[:2]
    cx, cy = w / 2, h / 2

    best = None
    best_score = -1

    for box, label in boxes:
        x1, y1, x2, y2 = box

        obj_cx = (x1 + x2) / 2
        obj_cy = (y1 + y2) / 2

        dist = np.sqrt((obj_cx - cx)**2 + (obj_cy - cy)**2)
        area = (x2-x1) * (y2-y1)

        score = area - 0.5 * dist  # tune weight

        if score > best_score:
            best_score = score
            best = (box, label)

    return best

def analyze_box(frame, box):
    x1, y1, x2, y2 = box

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    area = max(0, (x2 - x1) * (y2 - y1))

    h, w = frame.shape[:2]
    norm_area = area / (w * h)

    return {
        "center": (cx, cy),
        "area": area,
        "size_norm": norm_area,
        "depth_proxy": 1 - norm_area
    }

def compute_distance(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB

    acx = (ax1 + ax2) / 2
    acy = (ay1 + ay2) / 2

    bcx = (bx1 + bx2) / 2
    bcy = (by1 + by2) / 2

    return np.sqrt((acx - bcx)**2 + (acy - bcy)**2)

def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, (x2-x1) * (y2-y1))

def analyze_pl_objects(frame, boxes):
    if len(boxes) < 2:
        return None

    boxA, labelA = boxes[0]
    remaining = boxes[1:]

    resultB = pick_active_object(frame, remaining)
    if resultB is None:
        return None
    boxB, labelB = resultB

    a_info = analyze_box(frame, boxA)
    b_info = analyze_box(frame, boxB)

    dist = compute_distance(boxA, boxB)

    return {
        "boxA": boxA,
        "boxB": boxB,
        "A_label": labelA,
        "B_label": labelB,
        "A_meta": a_info,
        "B_meta": b_info,
        "distance": dist
    }