import cv2
import pytesseract
import numpy as np
import time
import pandas as pd
import re
import requests
import Levenshtein
import speech_recognition as sr

# Tesseract OCR path setup
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Dataset loading
df = pd.read_csv("E:/PROJECTS AND WORKS/PROJECT IDEA SUBMISSION/SIH'24 MEDBOT/medical_dataset.csv")

# OpenAI API function for searching online
def query_openai_api(message_content):
    api_key = 'your_api_key_here'  # Replace with your OpenAI API key
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": message_content}],
        "max_tokens": 100
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()['choices'][0]['message']['content']
        print(result)
        return result
    else:
        print("Failed to retrieve information from OpenAI.")
        return None

# Conversation handling for the medicine
def handle_conversation(medicine_name):
    print(f"You can now ask additional questions about {medicine_name}. Type 'exit' to end the conversation.")
    while True:
        user_query = input("Ask a question: ")
        if user_query.lower() == 'exit':
            print("Ending conversation. Goodbye!")
            break
        else:
            query_openai_api(f"Regarding {medicine_name}, {user_query}")

# Text preprocessing function
def clean_extracted_text(text):
    text = text.strip()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

# Levenshtein similarity for medicine name matching
def closest_match_levenshtein(extracted_text, medical_names):
    similarities = [Levenshtein.ratio(extracted_text, name) for name in medical_names]
    max_similarity_index = np.argmax(similarities)
    return max_similarity_index, similarities[max_similarity_index]

# Improved preprocessing the frame for better OCR
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Denoise using a median filter
    denoised = cv2.medianBlur(gray, 3)
    # Adaptive thresholding for better text visibility
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresh

# Extracting text from the live video feed (only when the correct text is detected)
def extract_text_from_live_video():
    cap = cv2.VideoCapture(0)  # Open the webcam
    print("Capturing live video. Press 'q' to stop early.")

    frame_skip = 5
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip some frames to improve processing time
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # Display the live video feed
        cv2.imshow('Live Video Feed', frame)

        # Preprocess the frame to improve OCR results
        processed_frame = preprocess_frame(frame)

        # Extract text from the processed frame
        extracted_text_frame = pytesseract.image_to_string(processed_frame, config='--psm 6 --oem 3')

        # Clean the extracted text
        cleaned_text = clean_extracted_text(extracted_text_frame)

        # Condition to check if the text is valid and matches the dataset
        if len(cleaned_text) > 2:

            # Check if the extracted text matches any medicine name in the dataset
            index, similarity_score = closest_match_levenshtein(cleaned_text, df['Medicine Name'].values)

            if similarity_score > 0.5:  # Found a valid match
                cap.release()
                cv2.destroyAllWindows()
                return df.iloc[index], similarity_score  # Return the medicine data and similarity score

        # Stop capturing if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return None, None

# Extract text from live audio (speech recognition)
def extract_text_from_live_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for 10 seconds... Speak now!")
        try:
            audio = recognizer.record(source, duration=10)  # Limit recording to 10 seconds
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return "Unable to recognize speech."
        except sr.WaitTimeoutError:
            return "No speech detected."

# Main function to handle user input and process accordingly
def main():
    input_type = input("Enter 'audio', 'video', or 'text': ").strip().lower()

    if input_type == 'audio':
        extracted_text = extract_text_from_live_audio()
        similarity_score = None

    elif input_type == 'video':
        extracted_text, similarity_score = extract_text_from_live_video()

    elif input_type == 'text':
        extracted_text = input("Please type the name of the medicine: ").strip()
        extracted_text = clean_extracted_text(extracted_text)
        similarity_score = None

    else:
        print("Invalid input type. Please enter 'audio', 'video', or 'text'.")
        return

    if extracted_text is not None:
        if similarity_score is None:  # For text or audio input, calculate similarity
            index, similarity_score = closest_match_levenshtein(extracted_text, df['Medicine Name'].values)
            extracted_text = df.iloc[index] if similarity_score > 0.5 else None

        if similarity_score > 0.5:
            medicine_name = extracted_text['Medicine Name']
            print(f"Medicine Found: {medicine_name}")
            print(extracted_text)
            print(f"Similarity Score: {similarity_score:.2f}")
            handle_conversation(medicine_name)
        else:
            print("No close match found. Searching online...")
            result = query_openai_api(
                f"Please provide information about the medicine '{extracted_text}' including its uses, side effects, and composition.")

            if result:
                print("Online Search Result:")
                print(result)
                handle_conversation(extracted_text)
    else:
        print("No text could be extracted.")

if __name__ == "__main__":
    main()
