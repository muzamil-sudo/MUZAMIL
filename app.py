import os
import PIL
import numpy as np
import google.generativeai as genai
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from dotenv import load_dotenv
from warnings import filterwarnings
filterwarnings(action='ignore')

class Whiteboard:

    def streamlit_config(self):
        # page configuration
        st.set_page_config(page_title='Whiteboard', layout="wide")

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
        st.markdown(f'<h1 style="text-align: center;">AIR-M Whiteboard</h1>',
                    unsafe_allow_html=True)
        add_vertical_space(1)

    def __init__(self):
        # Load the Env File for Secret API Key
        load_dotenv()

        # Initialize previous position for drawing
        self.p1, self.p2 = 0, 0

        # Initialize a blank canvas for drawing
        self.imgCanvas = np.zeros(shape=(550, 950, 3), dtype=np.uint8)

    def process_drawing(self, drawing_position):
        # If there is a drawing position (mouse click), draw on the canvas
        if drawing_position:
            cx, cy = drawing_position
            if self.p1 == 0 and self.p2 == 0:
                self.p1, self.p2 = cx, cy
            cv2.line(img=self.imgCanvas, pt1=(self.p1, self.p2), pt2=(cx, cy), color=(255, 0, 255), thickness=5)
            self.p1, self.p2 = cx, cy

    def handle_drawing_reset(self):
        # Reset the canvas when the clear button is clicked
        self.imgCanvas = np.zeros(shape=(550, 950, 3), dtype=np.uint8)

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
            # Streamlit drawing canvas widget (for mouse input)
            drawing_position = st.slider("Draw on the whiteboard (use mouse)", 0, 100, step=1)

        with col3:
            # Placeholder for result output
            st.markdown(f'<h5 style="text-align:center;color:green;">OUTPUT:</h5>', unsafe_allow_html=True)
            result_placeholder = st.empty()

        # Process mouse drawing
        self.process_drawing(drawing_position)

        # Display the drawn content
        st.image(self.imgCanvas, channels="RGB")

        # Button to trigger AI analysis
        if st.button("Analyze Drawing with AI"):
            result = self.analyze_image_with_genai()
            result_placeholder.write(f"Result: {result}")

        # Button to clear the canvas
        if st.button("Clear Canvas"):
            self.handle_drawing_reset()

try:
    # Create an instance of the Whiteboard class
    wb = Whiteboard()

    # Streamlit Configuration Setup
    wb.streamlit_config()

    # Calling the main method
    wb.main()

except Exception as e:
    # Displaying any error messages
    add_vertical_space(5)
    st.markdown(f'<h5 style="text-align:center;color:orange;">{e}</h5>', unsafe_allow_html=True)
