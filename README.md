> Note (To Dr. OH and Dr. OH alone: The original commit was made on Wednesday, April 29th, 2026. I've had to edit the project to 1) add a more beginner friendly READ_ME and 2) change the object detection model from MediaPipe to YOLOv8, which involved a lot more, and bigger files. GitHub wouldn't let me push my changes, so I had to delete the repo and start from scratch with the same name.
> 
> _TLDR: I did in fact complete the project before the deadline, and not on May 2nd, 2026 when the github republished._
---
# 🧠 Build-A-Brain: PSYC 203 Final Project, Spring 2026
An interactive computer vision + gesture-based learning game that teaches the brain’s lobes using real-time camera tracking, object detection, and facial/gesture analysis. 

Modules Involved:  **Module 2 - Mind & Brain** and **Module 4 - Perceiving and Recognizing Objects**

---
## Instructions (no coding experience needed)
### Step 1: Install Python (if you don't already have it installed)
Download and install Python (version 3.9 or newer):
+ https://www.python.org/downloads/

**IMPORTANT:**  
During installation, check the box that says:
✔ "Add Python to PATH"

### Step 2: Copy the Github link to this project
+ Click the green **"Code"** button on GitHub → click **"Copy URL to clipboard"**

### Step 3: Clone the Project's Repository
- **Mac:** Open “Terminal”  
- **Windows:** Open “Command Prompt”

**IMPORTANT:**  
Make sure you're in a folder that you *want* the project to be cloned into. you might have to cd into the correct folder, like Downloads or Desktop.

Type this (then press Enter):
`git clone [the-link-you-just-copied]`

### Step 4: Make sure all required libraries are installed
`cd build-a-brain-3
pip install -r requirements.txt`

### Step 5: Run the program!
`python main.py`

--
## What Should Happen:
+ A camera window will open
+ A slideshow window will appear
+ You’ll interact using:
 + hand gestures 👍 👎
 + objects (like holding items)
 + facial expressions
 + voice (when prompted)

## If Something Doesn’t Work
### Camera not showing?
+ Make sure your webcam is connected
+ Try changing this line in main.py:
`cam = cv2.VideoCapture(0)`

### “Module not found” error?
+ Run again:
`pip install -r requirements.txt`

### Mic not working?
+ Allow microphone permissions in system settings

## Tips
+ Good lighting helps detection a LOT
+ Hold objects clearly in front of the camera
+ Avoid cluttered backgrounds
_**🧠 That’s it!**_

If you made it this far, the game should be running 🎉

---
## Video Demo
<https://google.com>

--
## Features
+ Real-time object detection (YOLO-based)
+ Gesture tracking (hand input)
+ Facial expression detection
+ Voice input interaction
+ Interactive brain lobe learning system:
  + Occipital (vision)
  + Parietal (spatial reasoning)
  + Temporal (emotions/sound)
  + Frontal (speech/executive function)
+ Slide-based progression system with activity checkpoints

--
## Citations

### Title / UI Design
- Brain Background used in "Click-to-Play" text  
  Source: Pinterest  
  https://www.pinterest.com/pin/2322237302740203/  
  *Note: Title design created by Abena Poku; background image not owned*

### Brain Anatomy & Function
- Dana Foundation (2024). *Brain anatomy and function diagram*  
  https://dana.org/app/uploads/2023/09/anatomy-function-brain-areas-basics-aug-2019-2024.jpeg  

- Villines, Z. (2017, June 29). *What does the frontal lobe do?* Medical News Today  
  https://www.medicalnewstoday.com/articles/318139  

- Cleveland Clinic. *Parietal lobe: What it is, function, location & damage*  
  https://my.clevelandclinic.org/health/body/24628-parietal-lobe  

- Cleveland Clinic. *Temporal lobe: What it is, function, location & damage*  
  https://my.clevelandclinic.org/health/body/16799-temporal-lobe  

- Cleveland Clinic. *Occipital lobe: What it is, function, location & conditions*  
  https://my.clevelandclinic.org/health/body/24498-occipital-lobe  

- Stewart, S., & Sendić, G. (2023). *Lobes of the brain*. Kenhub  
  https://www.kenhub.com/en/library/anatomy/lobes-of-the-brain  

### Media & Assets
- GIFs: Various sources (Pinterest, untracked)
- Animal comparison visuals: Facebook Brain Maze (uncredited)
- Additional lobe visuals: Multiple online educational sources (untracked)

--
## Credits

Developed by **Abena Poku**

This project combines:
- Computer Vision (OpenCV, YOLO)
- Gesture & Face Tracking (MediaPipe)
- Speech Recognition
- Cognitive Science principles (PSYC 203 with Dr. OH at Rice University)
