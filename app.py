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


class Calculator:

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
        # Load the Env File for Secret API Key
        load_dotenv()

        # Initialize MediaPipe Hands object
        self.mphands = hands.Hands(max_num_hands=1, min_detection_confidence=0.75)

        # Set Drawing Origin to Zero
        self.p1, self.p2 = 0, 0

        # Set Previous Time for FPS calculation
        self.p_time = 0

        # Create Fingers Open/Close Position List
        self.fingers = []

        # Initialize a blank canvas
        self.imgCanvas = np.zeros(shape=(550, 950, 3), dtype=np.uint8)

    def process_frame(self, frame):
        # Flip the Image Horizontally for a Later Selfie_View Display
        self.img = cv2.flip(src=frame, flipCode=1)

        # Convert BGR Image to RGB Image
        self.imgRGB = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

    def process_hands(self):
        # Processes an RGB Image and Returns Hand Landmarks and Handedness
        result = self.mphands.process(image=self.imgRGB)

        # Draws the landmarks and connections on the image
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
                # Index, Middle, Ring, and Pinky Fingers
                if id != 4:
                    if self.landmark_list[id][2] < self.landmark_list[id-2][2]:
                        self.fingers.append(1)
                    else:
                        self.fingers.append(0)
                # Thumb Finger
                else:
                    if self.landmark_list[id][1] < self.landmark_list[id-2][1]:
                        self.fingers.append(1)
                    else:
                        self.fingers.append(0)

            for i in range(0, 5):
                if self.fingers[i] == 1:
                    cx, cy = self.landmark_list[(i+1)*4][1], self.landmark_list[(i+1)*4][2]
                    cv2.circle(img=self.img, center=(cx, cy), radius=5, color=(255, 0, 255), thickness=1)

    def handle_drawing_mode(self):
        # Handle various finger gestures for drawing, erasing, etc.
        if sum(self.fingers) == 2 and self.fingers[0] == self.fingers[1] == 1:
            cx, cy = self.landmark_list[8][1], self.landmark_list[8][2]
            if self.p1 == 0 and self.p2 == 0:
                self.p1, self.p2 = cx, cy
            cv2.line(img=self.imgCanvas, pt1=(self.p1, self.p2), pt2=(cx, cy), color=(255, 0, 255), thickness=5)
            self.p1, self.p2 = cx, cy
        elif sum(self.fingers) == 3 and self.fingers[0] == self.fingers[1] == self.fingers[2] == 1:
            self.p1, self.p2 = 0, 0
        elif sum(self.fingers) == 2 and self.fingers[0] == self.fingers[4] == 1:
            cx, cy = self.landmark_list[12][1], self.landmark_list[12][2]
            if self.p1 == 0 and self.p2 == 0:
                self.p1, self.p2 = cx, cy
            cv2.line(img=self.imgCanvas, pt1=(self.p1, self.p2), pt2=(cx, cy), color=(0, 0, 0), thickness=15)
            self.p1, self.p2 = cx, cy
        elif sum(self.fingers) == 2 and self.fingers[0] == self.fingers[4] == 1:
            self.imgCanvas = np.zeros(shape=(550, 950, 3), dtype=np.uint8)

    def blend_canvas_with_feed(self):
        img = cv2.addWeighted(src1=self.img, alpha=0.7, src2=self.imgCanvas, beta=1, gamma=0)
        imgGray = cv2.cvtColor(self.imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(src=imgGray, thresh=50, maxval=255, type=cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(src1=img, src2=imgInv)
        self.img = cv2.bitwise_or(src1=img, src2=self.imgCanvas)

    def analyze_image_with_genai(self):
        imgCanvas = cv2.cvtColor(self.imgCanvas, cv2.COLOR_BGR2RGB)
        imgCanvas = PIL.Image.fromarray(imgCanvas)
        genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
        model = genai.GenerativeModel(model_name='gemini-1.5-flash')
        prompt = "Analyze the image and provide the following:\n" \
                 "* The mathematical equation represented in the image.\n" \
                 "* The solution to the equation.\n" \
                 "* A short and sweet explanation of the steps taken to arrive at the solution."
        try:
            response = model.generate_content([prompt, imgCanvas])
            return response.text
        except Exception as e:
            return f"Error in AI response: {e}"

    def main(self):
        col1, _, col3 = st.columns([0.8, 0.02, 0.18])

        with col1:
            # Streamlit webcam input widget
            frame = st.camera_input("Capture a frame")

        with col3:
            # Placeholder for result output
            st.markdown(f'<h5 style="text-align:center;color:green;">OUTPUT:</h5>', unsafe_allow_html=True)
            result_placeholder = st.empty()

        if frame:
            # Process captured frame
            self.process_frame(frame)
            self.process_hands()
            self.identify_fingers()
            self.handle_drawing_mode()
            self.blend_canvas_with_feed()

            # Display the Output Frame in Streamlit
            st.image(self.img, channels="RGB")

            # Trigger AI analysis if specific finger gesture is detected
            if sum(self.fingers) == 2 and self.fingers[1] == self.fingers[2] == 1:
                result = self.analyze_image_with_genai()
                result_placeholder.write(f"Result: {result}")


try:
    # Create an instance of the Calculator class
    calc = Calculator()

    # Streamlit Configuration Setup
    calc.streamlit_config()

    # Calling the main method
    calc.main()

except Exception as e:
    # Displaying any error messages
    add_vertical_space(5)
    st.markdown(f'<h5 style="text-align:center;color:orange;">{e}</h5>', unsafe_allow_html=True)
