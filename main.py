import json
import os
import csv
import datetime
import random
import ssl
import streamlit as st
from streamlit_chat import message  # Install streamlit-chat for chat bubbles
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
for design_item in design:
    for pattern in design_item['patterns']:
        patterns.append(pattern)
        tags.append(design_item['tag'])

# Train the model
X = vectorizer.fit_transform(patterns)
y = tags
classifier.fit(X, y)

# Chatbot response function
def chatbot_response(user_input):
    input_vector = vectorizer.transform([user_input])
    predicted_tag = classifier.predict(input_vector)[0]
    for design_item in design:
        if design_item['tag'] == predicted_tag:
            return random.choice(design_item['responses'])

# Counter for input keys
user_input_counter = 0

# Main application function
def main():
    global user_input_counter
    st.set_page_config(page_title="Aura GPT", page_icon="💬", layout="wide")

    # Custom CSS for styling
    st.markdown("""
        <style>
            body {
                background-color: #f9f9f9;
                font-family: 'Arial', sans-serif;
            }
            .chat-title {
                color: #4CAF50;
                font-size: 2rem;
                text-align: center;
                margin-top: 10px;
            }
            .footer {
                position: fixed;
                bottom: 0;
                width: 100%;
                text-align: center;
                font-size: 0.9rem;
                color: #888;
            }
        </style>
    """, unsafe_allow_html=True)

    # Title and Header
    st.markdown("<h1 class='chat-title'>Aura GPT: Your Friendly Chatbot</h1>", unsafe_allow_html=True)

    # Sidebar menu
    menu = ["Chat", "History", "About"]
    menu_choice = st.sidebar.selectbox("Navigate", menu)

    if menu_choice == "Chat":
        st.write("### Welcome! Type your question below:")
        log_file = 'auralogs.csv'

        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        user_input_counter += 1
        user_message = st.text_input("You:", key=f"user_input_{user_input_counter}")

        if user_message:
            chatbot_reply = chatbot_response(user_message)
            message(user_message, is_user=True)  # User message bubble
            message(chatbot_reply)  # Chatbot message bubble

            # Log the interaction
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_file, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([user_message, chatbot_reply, timestamp])

            if chatbot_reply.lower() in ['goodbye', 'bye']:
                st.write("### Thank you for chatting! Have a wonderful day!")
                st.stop()

    elif menu_choice == "History":
        st.header("Conversation History")
        try:
            with open('auralogs.csv', 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # Skip header
                for row in reader:
                    st.markdown(f"**User**: {row[0]}")
                    st.markdown(f"**Chatbot**: {row[1]}")
                    st.markdown(f"**Time**: {row[2]}")
                    st.markdown("---")
        except FileNotFoundError:
            st.write("No conversation history found.")

    elif menu_choice == "About":
        st.subheader("Aura Overview")
        st.write("""
        Aura GPT is a conversational AI chatbot designed to understand user intents and provide meaningful responses.
        Built with NLP and machine learning, it leverages logistic regression for intent classification.
        """)
        st.subheader("Features")
        st.write("""
        - **Natural Language Processing**: Understands user input to identify intents.
        - **Streamlit Interface**: User-friendly and interactive.
        - **Conversation Logs**: Records chats for reference.
        """)

    # Footer
    st.markdown("<div class='footer'>Developed with ❤️ using Streamlit</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
