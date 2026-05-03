import time
"""
GOAL: GameState, GameController Classes, other gameplay, and control slides 
"""

class GameState:
    def __init__(self):
        self.mode = "intro"
        self.mode_start_time = time.time()
        self.mode_duration = 0

        self.activity = None
        self.models = {
            "face": False,
            "object": False,
            "voice": False
        }
        self.session_log = {
            "start_time": time.time(),
            "slides": [],
        }

        self.slide = 0
        self.input_mode = "voice"

        self.voice_response = None
        self.recording_voice = False
        self.voice_done = False

        self.expression_start_time = None
        self.expression_active = False
        self.expression_locked = None
        self.expression_prompt_shown = False

        self.region_lock = None
        self.region_hold_frames = 0
        self.input_lock_until = 0

        self.color_locked = False
        self.color_hold_frames = 0

        self.min_slide_time = 3.0
        self.slide_start_time = time.time()

        self.pl_hold_frames = 0
        self.pl_triggered = False
        self.pl_start_time = 0

        self.ol_logged = False
        self.pl_logged = False
        self.tl_logged = False
        self.fl_logged = False

        self.game_over = False

class GameController:
    def __init__(self, state, slides_ref):
        self.state = state
        self.slides = slides_ref

    def update_slide(self, new_slide):
        if new_slide not in self.slides:
            return
        
        config = slides[new_slide]
        self.state.slide = new_slide
        self.state.mode = config["mode"]
        self.state.activity = config["activity"]
        self.state.models = config["models"]
        self.state.slide_start_time = time.time()
        self.state.input_lock_until = time.time() + 1
        self.state.region_lock = None
        self.state.region_hold_frames = 0
        self.state.pl_triggered = False
        self.state.ol_logged = False
        self.state.pl_logged = False
        self.state.tl_logged = False
        self.state.fl_logged = False

        new_slide_data = self.slides[new_slide]

        if self.state.mode != new_slide_data["mode"]:
            self.state.mode_duration = time.time() - self.state.mode_start_time
            self.state.mode = new_slide_data["mode"]
            self.state.mode_start_time = time.time()


#lovely hooray input finally yas
    def handle_input(self, input_value):
        #handle commands from vision
        current_slide = self.state.slide
        slide_info = self.slides[current_slide]
        activity = slide_info["activity"]
        if input_value == "next_slide" and (activity != "activity" or current_slide in ACTIVITY_COOLDOWN):
            self.update_slide(current_slide + 1)
            return
        if input_value == "skip_two" and (activity == "activity" and current_slide not in ACTIVITY_COOLDOWN):
            self.update_slide(current_slide + 2)
            return
        
        # text or voice result (name and last activity slides)
        #um so i got rid of name thingy bc pointing to lobe way too fancy lmao no more keyboard i guess
        #voice thing for slide 29 alrdy handled in main.py
            

#slides!
def model_details(mode, activity):
    models = {"face": False, "object": False, "voice": False}
    
    if activity == "activity":
        if mode in ["ol", "pl"]:
            return {"face": False, "object": True, "voice": False}
        if mode == "tl":
            return {"face": True, "object": False, "voice": False}
        if mode == "fl":
            return {"face": False, "object": False, "voice": True}

    return models

ORDER = [ ("intro", "slide"),

    ("ol", "location"),
    ("ol", "slide"),
    ("ol", "slide"),
    ("ol", "slide"),
    ("ol", "activity"),
    ("ol", "activity"),

    ("pl", "location"),
    ("pl", "slide"),
    ("pl", "slide"),
    ("pl", "slide"),
    ("pl", "activity"),
    ("pl", "activity"),

    ("tl", "location"),
    ("tl", "slide"),
    ("tl", "slide"),
    ("tl", "slide"),
    ("tl", "activity"),
    ("tl", "activity"),

    ("fl", "location"),
    ("fl", "slide"),
    ("fl", "slide"),
    ("fl", "slide"),
    ("fl", "slide"),  # fl has 4 slides
    ("fl", "activity"),
    ("fl", "activity"),

    ("outro", "slide"),
]

slides = {}
for idx, (mode, activity) in enumerate(ORDER):
    slide_id = idx + 1
    slides[idx] = {
        "mode": mode,
        "activity": activity,
        "models": model_details(mode, activity),
        "type": "video" if activity == "location" else "image",
        "path": f"BaB_slides/{slide_id}" + (".mov" if activity == "location" else ".png")
        }

POINT_PROMPTS = {
    0: "Point to the occipital lobe",
    1: "Point to the parietal lobe",
    2: "Point to the temporal lobe",
    3: "Point to the frontal lobe"
}

ACTIVITY_COOLDOWN = [6, 12, 18, 25]