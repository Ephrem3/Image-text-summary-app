
import cv2
import streamlit as st
import easyocr as ocr
import google.generativeai as genai
import os
import itertools
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()

# Function to capture image using OpenCV and Streamlit
def capture_image():
    try:
        cap = cv2.VideoCapture(0)  # Change the index to 1 for the external camera
        if not cap.isOpened():
            st.error("Failed to open external camera. Please ensure that the camera is connected.")
            return None

        ret, frame = cap.read()
        cap.release()

        if not ret:
            st.error("Failed to capture frame from external camera.")
            return None

        return frame
    except Exception as e:
        st.error(f"An error occurred while capturing image from external camera: {e}")
        return None



# Function to perform OCR on the captured image using EasyOCR
def perform_ocr(image):
    reader = ocr.Reader(['en'], gpu=True)
    result = reader.readtext(image)
    return result

# Function to generate summary using GenerativeAI
def generate_summary(text):
    try:
        prompt = text + '\nsummarize'  # Add "summarize" at the end of the text for summary generation
        genai.configure(api_key=os.getenv('API_KEY'))
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def main():
    # Streamlit UI
    st.title("Image Text Summary App")

    # Display live camera feed
    st.write("Live Camera Feed:")
    frame_placeholder = st.empty()
    cap = cv2.VideoCapture(0)
    
    # Counter for the button key
    button_counter = itertools.count()

    # Capture Image button with unique key
    capture_button_key = f"capture_button_{next(button_counter)}"
    capture_button_pressed = st.button("Capture Image", key=capture_button_key)

    if capture_button_pressed:
        st.write("Capturing Image...")
        frame = capture_image()
        #st.image(frame, channels="BGR", caption="Captured Image")

        st.write("Processing Image...")
        # Resize image for better OCR performance
        resized_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        ocr_result = perform_ocr(resized_frame)
        
        # Draw bounding boxes and recognized words on the image
        for detection in ocr_result:
            box, text = detection[:2]
            top_left = tuple(map(int, box[0]))
            bottom_right = tuple(map(int, box[2]))
            # Draw rectangle and text on the image
            resized_frame = cv2.rectangle(resized_frame, top_left, bottom_right, (0, 255, 0), 3)
            resized_frame = cv2.putText(resized_frame, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 10), 2, cv2.LINE_AA)
        
        st.image(resized_frame, channels="BGR", caption="Processed Image")

        extracted_text = '\n'.join(detection[1] for detection in ocr_result)
        st.write("Extracted Text:")
        st.write(extracted_text)

        st.write("Generating Summary...")
        summary = generate_summary(extracted_text)
        st.write("Summary:")
        st.write(summary)
        
        st.write("Press the button below to capture another image.")
        reset_button_key = f"reset_button_{next(button_counter)}"
        reset_button_pressed = st.button("Capture Another Image", key=reset_button_key)
        
        if reset_button_pressed:
            st.experimental_rerun()

    else:
        while True:
            ret, frame = cap.read()
            if ret:
                frame_placeholder.image(frame, channels="BGR", caption="Live Camera Feed")
            else:
                st.write("Failed to capture frame")
                break

if __name__ == "__main__":
    main()
