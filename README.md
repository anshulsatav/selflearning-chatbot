# selflearning-chatbot

This is an intelligent chatbot application built using Python, TensorFlow, and NLTK. It can respond to user inputs, provide weather information, stock prices, and learn from user feedback to improve its responses over time.

# Features
1. Intent Recognition: Uses a trained neural network model to predict the intent of user inputs.
2. Weather Information: Fetches and displays weather information for a specified location and date range using the Visual Crossing Weather API.
3. Stock Prices: Provides the latest stock prices for popular companies using the Alpha Vantage API.
4. Feedback and Retraining: Collects user feedback on responses and retrains the model to improve its accuracy.

# User Interface: Simple GUI built using Tkinter.
# Requirements
1. Python 3.6+
2. TensorFlow
3. NLTK
4. Requests
5. Tkinter

# File Structure
1. chatbot.py: Main application script.
2. intents.json: JSON file containing predefined intents and responses.
3. chatbot/words.pkl: Pickle file containing the words used in training.
4. chatbot/classes.pkl: Pickle file containing the classes (intents) used in training.
5. chatbot/chatbot_model.keras: Trained model file.
6. feedback/feedback_data.json: JSON file where user feedback is saved

# Usage
1. Weather Information:

To get weather information, type a message in the format: weather [location] [date1] [date2]
Example: weather London 2023-08-01 2023-08-05

2. Stock Prices:

To get stock prices, type a message in the format: stock price of [company name]
Example: stock price of Apple

3. Chat and Feedback:

Type any other message to chat with the bot.
After receiving a response, the bot will ask for feedback (good or bad). This feedback is used to retrain the model.

4. Set Up API Keys:

Visual Crossing Weather API: Replace "enter your api key" in the get_weather_report function with your actual API key.
Alpha Vantage API: Replace "enter your api key" in the get_stock_data function with your actual API key.

# Clone the Repository:
git clone https://github.com/anshulsatav/selflearning-chatbot.git
cd intelligent-chatbot

