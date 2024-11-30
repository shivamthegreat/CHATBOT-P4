#importing all required modules(libraries)
import json
import os
import csv
import datetime
import random
import ssl
import streamlit as st
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Configure SSL to avoid download errors for NLTK
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load design data from the JSON file
filedata = os.path.abspath("./design.json")
with open(filedata, "r") as file:
    design = json.load(file)

# Initialize the TF-IDF vectorizer and logistic regression classifier
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
classifier = LogisticRegression(random_state=0, max_iter=10000)

# Prepare data for training
tags = []
patterns = []
for design in design:
    for pattern in design['patterns']:
        patterns.append(pattern)
        tags.append(design['tag'])

# Train the model
X = vectorizer.fit_transform(patterns)
y = tags
classifier.fit(X, y)

# Chatbot response function
def chatbot_response(user_input):
    input_vector = vectorizer.transform([user_input])
    predicted_tag = classifier.predict(input_vector)[0]
    for design in design:
        if design['tag'] == predicted_tag:
            return random.choice(design['responses'])

# Counter for input keys
user_input_counter = 0

# Main application function
def main():
    global user_input_counter
    st.title("Chatbot Using NLP")

    # Sidebar menu
    menu = ["Dashboard", "History", "About"]
    menu_choice = st.sidebar.selectbox("Navigate", menu)

    if menu_choice == "Dashboard":
        st.write("Welcome! This is Aura_gpt Type your question...")

        # Ensure the chat log file exists
        log_file = 'auralogs.csv'
        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        user_input_counter += 1
        user_message = st.text_input("You:", key=f"user_input_{user_input_counter}")

        if user_message:
            chatbot_reply = chatbot_response(user_message)
            st.text_area("Chatbot:", value=chatbot_reply, height=150, max_chars=None, key=f"chatbot_reply_{user_input_counter}")

            # Log the interaction
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_file, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([user_message, chatbot_reply, timestamp])

            if chatbot_reply.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting! Have a wonderful day!")
                st.stop()

    elif menu_choice == "Conversation History":
        st.header("Previous Conversations")
        try:
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # Skip header
                for row in reader:
                    st.text(f"User: {row[0]}")
                    st.text(f"Chatbot: {row[1]}")
                    st.text(f"Time: {row[2]}")
                    st.markdown("---")
        except FileNotFoundError:
            st.write("No conversation history found.")

    elif menu_choice == "About":
        st.subheader("Aura Overview")
        st.write("""
        This project is a chatbot also called as Aura Gpt designed to identify user intents and respond accordingly.
        It leverages NLP techniques and machine learning, using logistic regression for intent classification.
        """)
        st.subheader("Features")
        st.write("""
        - **Learning the Data**: The chatbot is trained using labeled examples of user messages and their intents.  
        - **User-Friendly Design**: The interface, created with Streamlit, makes chatting simple and engaging.  
        - **Conversation Records**: Keeps a history of chats for easy reference and review.  
        """)
   





if __name__ == '__main__':
    main()
