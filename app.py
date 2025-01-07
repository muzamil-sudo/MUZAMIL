import os
import PIL
import numpy as np
import google.generativeai as genai
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from mediapipe.python.solutions import hands, drawing_utils
from dotenv import load_dotenv
from warnings import filterwarnings
filterwarnings(action='ignore')


class calculator:

    def streamlit_config(self):
        # page configuration
        st.set_page_config(page_title='Calculator', layout="wide")

        # page header transparent color and Removes top padding 
        page_background_color = """
        <style>

        [data-testid="stHeader"] 
        {
        background: rgba(0,0,0,0);
        }

        .block-container {
            padding-top: 0rem;
        }

        </style>
        """
        st.markdown(page_background_color, unsafe_allow_html=True)

        # title and position
        st.markdown(f'<h1 style="text-align: center;">AIR-M</h1>',
                    unsafe_allow_html=True)
        add_vertical_space(1)

    def __init__(self):
        # Load the Env File for Secrect API Key
        load_dotenv()

        # Initialize Canvas Image
        self.imgCanvas = np.zeros(shape=(550, 950, 3), dtype=np.uint8)

        # Initializes a MediaPipe Hand object
        self.mphands = hands.Hands(max_num_hands=1, min_detection_confidence=0.75)

        # Set Drawing Origin to Zero
        self.p1, self.p2 = 0, 0

        # Set Previous Time to Zero for FPS
        self.p_time = 0

        # Create Fingers Open/Close Position List
        self.fingers = []

    def process_frame(self, img):
        # Resize the Image
        img = cv2.resize(src=img, dsize=(950, 550))

        # Flip the Image Horizontally for Later Selfie_View Display
        self.img = cv2.flip(src=img, flipCode=1)

        # BGR Image Convert to RGB Image
        self.imgRGB = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

    def process_hands(self):
        # Processes an RGB Image and Returns the Hand Landmarks and Handedness of each Detected Hand
        result = self.mphands.process(image=self.imgRGB)

        # Draws the landmarks and the connections on the image
        self.landmark_list = []

        if result.multi_hand_landmarks:
            for hand_lms in result.multi_hand_landmarks:
                drawing_utils.draw_landmarks(image=self.img, landmark_list=hand_lms,
                                            connections=hands.HAND_CONNECTIONS)

                # Extract ID and Origin for Each Landmark
                for id, lm in enumerate(hand_lms.landmark):
                    h, w, c = self.img.shape
                    x, y = lm.x, lm.y
                    cx, cy = int(x * w), int(y * h)
                    self.landmark_list.append([id, cx, cy])

    def identify_fingers(self):
        # Identify Each Finger's Open/Close Position
        self.fingers = []

        if self.landmark_list != []:
            for id in [4, 8, 12, 16, 20]:
                if id != 4:
                    if self.landmark_list[id][2] < self.landmark_list[id - 2][2]:
                        self.fingers.append(1)
                    else:
                        self.fingers.append(0)
                else:
                    if self.landmark_list[id][1] < self.landmark_list[id - 2][1]:
                        self.fingers.append(1)
                    else:
                        self.fingers.append(0)

            # Identify Finger Open Position
            for i in range(0, 5):
                if self.fingers[i] == 1:
                    cx, cy = self.landmark_list[(i + 1) * 4][1], self.landmark_list[(i + 1) * 4][2]
                    cv2.circle(img=self.img, center=(cx, cy), radius=5, color=(255, 0, 255), thickness=1)

    def handle_drawing_mode(self):
        # Both Thumb and Index Fingers Up in Drawing Mode
        if sum(self.fingers) == 2 and self.fingers[0] == self.fingers[1] == 1:
            cx, cy = self.landmark_list[8][1], self.landmark_list[8][2]
            if self.p1 == 0 and self.p2 == 0:
                self.p1, self.p2 = cx, cy
            cv2.line(img=self.imgCanvas, pt1=(self.p1, self.p2), pt2=(cx, cy), color=(255, 0, 255), thickness=5)
            self.p1, self.p2 = cx, cy

        # Thumb, Index & Middle Fingers UP ---> Disable the Points Connection
        elif sum(self.fingers) == 3 and self.fingers[0] == self.fingers[1] == self.fingers[2] == 1:
            self.p1, self.p2 = 0, 0

        # Both Thumb and Middle Fingers Up ---> Erase the Drawing Lines
        elif sum(self.fingers) == 2 and self.fingers[0] == self.fingers[2] == 1:
            cx, cy = self.landmark_list[12][1], self.landmark_list[12][2]
            if self.p1 == 0 and self.p2 == 0:
                self.p1, self.p2 = cx, cy
            cv2.line(img=self.imgCanvas, pt1=(self.p1, self.p2), pt2=(cx, cy), color=(0, 0, 0), thickness=15)
            self.p1, self.p2 = cx, cy

        # Both Thumb and Pinky Fingers Up ---> Erase the Whole Thing (Reset)
        elif sum(self.fingers) == 2 and self.fingers[0] == self.fingers[4] == 1:
            self.imgCanvas = np.zeros(shape=(550, 950, 3), dtype=np.uint8)

    def blend_canvas_with_feed(self):
        # Blend the Live Camera Feed and Canvas Images ---> Canvas Image Top on it the Original Transparency Image
        img = cv2.addWeighted(src1=self.img, alpha=0.7, src2=self.imgCanvas, beta=1, gamma=0)

        # Canvas_BGR Image Convert to Gray Scale Image ---> Maintain Intensity of Color Image
        imgGray = cv2.cvtColor(self.imgCanvas, cv2.COLOR_BGR2GRAY)

        # Gray Image Convert to Binary_Inverse Image ---> Gray Shades into only Two Colors (Black/White) based Threshold
        _, imgInv = cv2.threshold(src=imgGray, thresh=50, maxval=255, type=cv2.THRESH_BINARY_INV)

        # Binary_Inverse Image Convert into BGR Image ---> Single Channel Value apply All 3 Channel [0,0,0] or [255,255,255]
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

        # Blending both Images ---> Binary_Inverse Image Black/White Top on Original Image
        img = cv2.bitwise_and(src1=img, src2=imgInv)

        # Canvas Color added on the Top on Original Image
        self.img = cv2.bitwise_or(src1=img, src2=self.imgCanvas)

    def analyze_image_with_genai(self):
        # Canvas_BGR Image Convert to RGB Image 
        imgCanvas = cv2.cvtColor(self.imgCanvas, cv2.COLOR_BGR2RGB)

        # Numpy Array Convert to PIL Image
        imgCanvas = PIL.Image.fromarray(imgCanvas)

        # Configures the genai Library
        genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

        # Initializes a Flash Generative Model
        model = genai.GenerativeModel(model_name='gemini-1.5-flash')

        # Input Prompt
        prompt = "Analyze the image and provide the following:\n" \
                 "* The mathematical equation represented in the image.\n" \
                 "* The solution to the equation.\n" \
                 "* A short and sweet explanation of the steps taken to arrive at the solution."

        # Sends Request to Model to Generate Content using a Text Prompt and Image
        response = model.generate_content([prompt, imgCanvas])

        # Extract the Text Content of the Model’s Response.
        return response.text

    def main(self):
        col1, _, col3 = st.columns([0.8, 0.02, 0.18])

        with col1:
            # Stream the webcam video
            video_input = st.camera_input("Capture Image")

            if video_input:
                img = np.array(PIL.Image.open(video_input))
                self.process_frame(img)
                self.process_hands()
                self.identify_fingers()
                self.handle_drawing_mode()
                self.blend_canvas_with_feed()

                # Display the Output Frame in the Streamlit App
                self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
                st.image(self.img, channels="RGB")

                # After Done Processing with AI
                if sum(self.fingers) == 2 and self.fingers[1] == self.fingers[2] == 1:
                    result = self.analyze_image_with_genai()
                    st.write(f"Result: {result}")

