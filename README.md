# CHATBOT-P4
This is my P4 internship project for Implementation of chat bot using NLP

# Aura Chatbot using NLP - A friendly chatbot

## Summary
This project involves the development of an intelligent chatbot using Natural Language Processing (NLP) to interact with users through text. The primary goal was to create a conversational agent capable of understanding user queries and providing relevant responses. The chatbot was built with the objective of enhancing user interaction by processing and interpreting language data efficiently.
---

## tech-Used
- **Python**
- **NLTK**
- **Scikit-learn**
- **Streamlit**
- **JSON** for intents data

---



### Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```


## To download the necessary NLTK data, use the following code:

'''python'''
import nltk
nltk.download('punkt')

 ## Install Requirements txt
```bash
pip install -r requirements.txt
```
## Usage
To run the chatbot application, execute the following command:
```bash
streamlit run app.py

```
On your terminal by using cmd and selecting the main file .
## working
Once the application is running, you can interact with the chatbot via the web interface. Simply type your message in the input box and press Enter to receive a response.  



## Chat History  
The chatbot stores conversation logs in a CSV file (`auralogs.csv`).
To access past interactions, use the "History" option available in the sidebar.  

---

## License  
This project is licensed under the General GitHub License.  

## Acknowledgments
*Aura Overview*
Aura GPT is a conversational AI chatbot designed to understand user intents and provide meaningful responses. Built with NLP and machine learning, it leverages logistic regression for intent classification.

## Features
Natural Language Processing: Understands user input to identify intents.
Streamlit Interface: User-friendly and interactive.
Conversation Logs: Records chats for reference.
## Thanks for using

---
