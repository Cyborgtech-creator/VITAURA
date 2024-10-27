import pandas as pd
from fuzzywuzzy import fuzz
import cv2
import numpy as np
import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox
import PyPDF2

# Load the CSV data
df = pd.read_csv('CSV/train.csv')

def check_similarity(input_issue, df):
    sim = 0
    r = None
    results = []
    
    for index, row in df.iterrows():
        similarity = fuzz.ratio(input_issue.lower(), row['Context'].lower())
        if sim < similarity:
            r = row
            sim = similarity
            
    if sim > 40:
        results.append({
            'Context': r['Context'],
            'similarity': sim,
            'Response': r['Response'],
        })
    
    return results

def chatbot(input_issue):
    matches = check_similarity(input_issue, df)
    response = ""
    if matches:
        for match in matches:
            response += f"(Similarity: {match['similarity']}%)\n"
            response += f"ECHO: {match['Response']}\n"
    else:
        response = "No similar issues found."
    
    return response

def predict_mood(frame):
    # Placeholder for mood prediction logic
    moods = ['happy', 'sad', 'neutral']
    return np.random.choice(moods)

def greet_user(mood):
    greetings = {
        'happy': "Hello! It's great to see you happy!",
        'sad': "Hello! I hope things get better for you soon.",
        'neutral': "Hello! How can I assist you today?",
    }
    return greetings.get(mood, "Hello! How can I assist you today?")

# Access the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Capture a frame and predict mood
ret, frame = cap.read()
if ret:
    mood = predict_mood(frame)
    greeting = greet_user(mood)
else:
    greeting = "Error: Could not read frame."

# Create the GUI
def on_enter(event=None):
    user_input = user_input_entry.get()
    if user_input.lower() in ['exit', 'quit']:
        root.quit()
    else:
        response = chatbot(user_input)
        chat_area.config(state=tk.NORMAL)
        chat_area.insert(tk.END, f"You: {user_input}\n")
        chat_area.insert(tk.END, f"Bot: {response}\n")
        chat_area.config(state=tk.DISABLED)
        user_input_entry.delete(0, tk.END)

def upload_text_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if file_path:
        with open(file_path, 'r') as file:
            content = file.read()
            user_input_entry.delete(0, tk.END)
            user_input_entry.insert(0, content)

def upload_pdf_file():
    file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
    if file_path:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            content = ""
            for page in reader.pages:
                content += page.extract_text() + "\n"
            user_input_entry.delete(0, tk.END)
            user_input_entry.insert(0, content)

def upload_image_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", ".jpg;.jpeg;*.png")])
    if file_path:
        comp(file_path)

def compare_images(imageA, imageB):
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    histA = cv2.calcHist([grayA], [0], None, [256], [0, 256])
    histB = cv2.calcHist([grayB], [0], None, [256], [0, 256])

    cv2.normalize(histA, histA)
    cv2.normalize(histB, histB)

    similarity = cv2.compareHist(histA, histB, cv2.HISTCMP_CORREL)
    similarity_percentage = similarity * 100

    return similarity_percentage

def comp(file_path):
    image1 = cv2.imread('Sample_reports/Hand.jpg')  # Replace with your image path
    image2 = cv2.imread(file_path)  # Replace with your image path

    if image1 is None or image2 is None:
        print("Error: One of the images could not be loaded.")
        chat_area.config(state=tk.NORMAL)
        chat_area.insert(tk.END, "Error: One of the images could not be loaded.\n")
        chat_area.config(state=tk.DISABLED)
    else:
        similarity = compare_images(image1, image2)
        if similarity > 5:
            chat_area.config(state=tk.NORMAL)
            # chat_area.insert(tk.END, f"Similarity: {similarity:.2f}%\n")
            chat_area.insert(tk.END, "You don't have any fractures.\n")
            chat_area.config(state=tk.DISABLED)
        else:
            chat_area.config(state=tk.NORMAL)
            # chat_area.insert(tk.END, f"Similarity: {similarity:.2f}%\n")
            chat_area.insert(tk.END, "You have a fracture. It is preferred you consult a doctor.\n")
            chat_area.config(state=tk.DISABLED)

# Set up the main application window
root = tk.Tk()
root.title("ECHO")

# Create a scrollable text area for chat display
chat_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED)
chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Display the greeting
chat_area.config(state=tk.NORMAL)
chat_area.insert(tk.END, f"Bot: {greeting}\n")
chat_area.config(state=tk.DISABLED)

# Entry for user input
user_input_entry = tk.Entry(root, width=50)
user_input_entry.pack(padx=10, pady=10, side=tk.LEFT, fill=tk.X, expand=True)

# Enter button
enter_button = tk.Button(root, text="Enter", command=on_enter)
enter_button.pack(padx=10, pady=10, side=tk.RIGHT)

# Upload buttons
upload_text_button = tk.Button(root, text="Upload Text File", command=upload_text_file)
upload_text_button.pack(padx=10, pady=10, side=tk.LEFT)

upload_pdf_button = tk.Button(root, text="Upload PDF File", command=upload_pdf_file)
upload_pdf_button.pack(padx=10, pady=10, side=tk.LEFT)

upload_image_button = tk.Button(root, text="Upload Image File", command=upload_image_file)
upload_image_button.pack(padx=10, pady=10, side=tk.LEFT)

# Bind the Enter key to the on_enter function
user_input_entry.bind('<Return>', on_enter)

# Run the GUI main loop
root.mainloop()

# Release the camera
cap.release()
cv2.destroyAllWindows()