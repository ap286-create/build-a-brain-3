import json
import cv2
import numpy as np
import mediapipe as mp
import time

import gameplay
import vision

import threading
import speech_recognition as sr


"""
GOAL: to run the camera loop and rendering
"""


current_frame = None
annotated_frame = None

mode = "intro"
active_image = None

hand_landmarks = None


cam = cv2.VideoCapture(1)
slides = None
current_slide_index = -1

#make slideshow window float on top always
slide_window = "Build-A-Brain Game"
cv2.namedWindow(slide_window, cv2.WINDOW_NORMAL)

#shrinking camera window size
camera_window = "Camera"
cv2.namedWindow(camera_window, cv2.WINDOW_NORMAL)
cv2.moveWindow(camera_window, 1039, 0)
cv2.setWindowProperty(camera_window, cv2.WND_PROP_TOPMOST, 1)

#pulling in other modules ;P
state = gameplay.GameState()
controller = gameplay.GameController(state, gameplay.slides)
vision_system = vision.VisionSystem()

#make sure window for slide acc shows slide and can be updated according to output from vision
def load_slide(index):
    global slides, current_slide_index
    if index not in gameplay.slides:
        return
    path = gameplay.slides[index]["path"]
    if slides is not None:
        slides.release()
    slides = cv2.VideoCapture(path)
    current_slide_index = index

#keyboard and voice and point event listeners
def start_voice_recording(state, controller):
    def record():
        print("[VOICE] started recording")

        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("[VOICE] listening...")
            audio = r.listen(source)

        print("[VOICE] audio captured")

        try:
            text = r.recognize_google(audio)
            print("[VOICE] recognized:", text)
            state.voice_response = text
        except Exception as e:
            print("[VOICE ERROR]", e)
            state.voice_response = "Could not understand audio"

    state.recording_voice = False
    state.voice_done = True
    print("[VOICE] done flag set")
    threading.Thread(target=record, daemon=True).start()

def get_region(x, y):
    if x < 0.5 and y < 0.5:
        return 0
    if x >= 0.5 and y < 0.5:
        return 1
    if x < 0.5 and y >= 0.5:
        return 2
    return 3
    

#cleanup
def end_game(state):
    print("\n===== SESSION COMPLETE =====")

    total_time = time.time() - state.session_log["start_time"]
    state.session_log["total_time"] = total_time

    print(f"\nTotal time: {round(total_time, 2)} seconds\n")

    for i, entry in enumerate(state.session_log["slides"]):
        print(f"\n--- Event {i+1} ---")
        for k, v in entry.items():
            print(f"{k}: {v}")

    print("\n===========================\n")

    # still save file
    with open("results.txt", "w") as f:
        json.dump(state.session_log, f, indent=2)

    cam.release()
    if slides is not None:
        slides.release()
    cv2.destroyAllWindows()

#MAIN LOOP LESGOOO
try:
    while True:
        #"y r we still running when it's over"
        if state.game_over:
            break
        #deciding whether video or photo for slideshow
        slide = gameplay.slides[state.slide]
        if slide["type"] == "image":
            frame_slides = cv2.imread(slide["path"])
        elif slide["type"] == "video":
            if current_slide_index != state.slide:
                load_slide(state.slide)
                
            hasFrame, frame_slides = slides.read()

            if not hasFrame:
                slides.set(cv2.CAP_PROP_POS_FRAMES, 0)
                hasFrame, frame_slides = slides.read()

        #is this that one slide where u gotta speak?
        if slide["activity"] == "activity" and slide["mode"] == "fl":
            if not state.recording_voice:
                print("Talk about why you chose this object...")
                state.recording_voice = True
                start_voice_recording(state, controller)
            if state.voice_done and not state.fl_logged:
                controller.update_slide(state.slide + 1)
                #loggg
                state.session_log["slides"].append({
                    "event": "Frontal Lobe Activity",
                    "voice_response": state.voice_response,
                    "time_spent": time.time() - state.mode_start_time})
                state.fl_logged = True
                state.voice_done = False
                state.recording_voice = False
                

        
        # is the slide with the expression?
        if slide["activity"] == "activity" and slide["mode"] == "tl":
            if not state.expression_prompt_shown:
                print("Make an expression...")
                state.expression_prompt_shown = True
                state.expression_start_time = None
                state.expression_active = False
                state.expression_locked = None

            # start timer once expression appears
            if not state.expression_active and vision_system.expression_label not in ["neutral", None]:
                state.expression_active = True
                state.expression_start_time = time.time()
                state.expression_locked = vision_system.expression_label

            # after 3 seconds → move on
            if state.expression_active:
                if time.time() - state.expression_start_time >= 3:
                    print("Expression detected:", state.expression_locked)
                    controller.update_slide(state.slide + 1)

                    #loggg
                    if not state.fl_logged:
                        state.session_log["slides"].append({
                            "event": "Temporal Lobe Activity",
                            "expression": state.expression_locked,
                            "time_spent": time.time() - state.mode_start_time})
                        state.fl_logged = True
                    
                    state.expression_prompt_shown = False
                    state.expression_start_time = None
                    state.expression_active = False
                    state.expression_locked = None
                    
                    
    
        #handle camera
        hasFrame_cam, frame_cam = cam.read()

        if not hasFrame_cam:
            break

        frame_cam = cv2.flip(frame_cam, 1)
        mp_image, display_frame = vision_system.prepare(frame_cam)
        vision_system.process(mp_image, frame_cam, state)
        
        #k render + display both
        display_frame = vision_system.render(display_frame)
        display_frame = cv2.resize(display_frame, (400, 225))

        #compute region for location slides
        # x = vision_system.pointer_x
        # y = vision_system.pointer_y
        # if x is not None and y is not None:
        #     region = get_region(x, y)
        # else:
        #     region = None

        #don't consider input that happens too early (3s slide rule) + gesture carryover
        now = time.time()
        allow_input = (now - state.slide_start_time >= state.min_slide_time 
                       and now >= state.input_lock_until)

        command = vision_system.get_command()
        if command and allow_input:
            controller.handle_input(command)

        # wait wait is this a mode=location slide?
        if slide["activity"] == "location":
            opts = ["Occipital", "Parietal", "Temporal", "Frontal"]
            correct = {
                "ol": 0,
                "pl": 1,
                "tl": 2,
                "fl": 3
            }.get(slide["mode"], -1)

            # draw labels
            cv2.putText(display_frame, opts[0], (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
            cv2.putText(display_frame, opts[1], (240, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
            cv2.putText(display_frame, opts[2], (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
            cv2.putText(display_frame, opts[3], (240, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

            x = vision_system.pointer_x
            y = vision_system.pointer_y
            if x is not None and y is not None:
                region = get_region(x, y)
            else:
                region = None

            if allow_input and region is not None:
            # -------- STABLE HOLD LOGIC --------
                if region == state.region_lock:
                    state.region_hold_frames += 1
                else:
                    state.region_lock = region
                    state.region_hold_frames = 0

                # require ~15 frames (~0.5s at 30fps)
                if state.region_hold_frames >= 15:

                    if region == correct:
                        print("Correct!")
                        controller.update_slide(state.slide + 1)
                        x = None
                        y = None
                    else:
                        print("Wrong")
                    # reset AFTER decision
                    state.region_lock = None
                    state.region_hold_frames = 0

        #or or or is this a ol activity or pl activity slide
        boxes = vision_system.get_boxes()

        ## (ol)
        if slide["mode"] == "ol" and slide["activity"] == "activity" and boxes:
            active, label = vision.pick_active_object(frame_cam, boxes)
            hex_color, bgr = vision.get_object_color_hex(frame_cam, active)

            #draw color dot on screen
            display_frame = vision.draw_color_indicator(display_frame, active, bgr)
            cv2.putText(display_frame, hex_color, (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, bgr, 2)
            
            if vision.redness_score(bgr) > 60:
                state.color_hold_frames += 1
            else:
                state.color_hold_frames = 0

            if state.color_hold_frames >= 15 and not state.color_locked:
                print("Red confirmed!")
                state.color_locked = True
                controller.update_slide(state.slide + 1)
                state.color_hold_frames = 0
                state.color_locked = False

                #loggg
                if not state.ol_logged:
                    y, obj = vision.pick_active_object(frame_cam, boxes)
                    col, x = vision.get_object_color_hex(frame_cam, active)
                    state.session_log["slides"].append({
                        "event": "Occipital Lobe Activity",
                        "color": col,
                        "object": obj, 
                        "time_spent": time.time() - state.mode_start_time})
                    state.ol_logged = True


        ## (pl)
        if slide["mode"] == "pl" and slide["activity"] == "activity":
            data = vision.analyze_pl_objects(frame_cam, boxes)

            if data is not None:
                dist = data["distance"]
                a_name = data["A_label"]
                b_name = data["B_label"]
                a_info = data["A_meta"]
                b_info = data["B_meta"]


                bigger = ""

                if data["boxA"] is not None and data["boxB"] is not None:
                    if a_info["area"] > b_info["area"]:
                        bigger = f"{a_name} is bigger!"  
                    else:
                        bigger = f"{b_name}is bigger!"

                cv2.putText(display_frame, f"Distance: {int(dist)}", (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                cv2.putText(display_frame, f"{a_name} depth: {a_info['depth_proxy']:.2f}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
                cv2.putText(display_frame, f"{b_name}depth: {b_info['depth_proxy']:.2f}", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
                cv2.putText(display_frame, bigger, (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

            if len(boxes) >= 2:
                state.pl_hold_frames += 1
            else:
                state.pl_hold_frames = 0
                state.pl_triggered = False

            real_activity = state.slide not in gameplay.ACTIVITY_COOLDOWN
            if state.pl_hold_frames >= 15 and real_activity and not state.pl_triggered:
                print("Two Objects confirmed!")
                controller.update_slide(state.slide + 1)
                state.pl_hold_frames = 0
                state.pl_triggered = True

                #loggg
                if not state.pl_logged:
                    state.session_log["slides"].append({
                        "event": "Parietal Lobe Activity",
                        "distance": dist,
                        "objects": f"{a_name} and {b_name}",
                        "time_spent": time.time() - state.mode_start_time
                    })
                    state.pl_logged = True

        #phew okay now we can show the windows hooray
        cv2.imshow(slide_window, frame_slides)
        cv2.imshow(camera_window, display_frame)

        if cv2.waitKey(1) != -1:
            state.game_over = True
except Exception as e:
    print("Error:", e)
finally: 
    end_game(state)

