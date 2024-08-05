# importing all the necessary modules
import random  
import json  
import pickle  
import numpy as np  
import nltk  
from nltk.stem import WordNetLemmatizer  
from tensorflow.keras.models import load_model  
import os  
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense, Dropout  
from tensorflow.keras.optimizers import SGD  
from tkinter import *  
import requests  

# Download required NLTK files
nltk.download('punkt')  
nltk.download('wordnet')  

# Initialize the word lemmatizer(word-> base form)
lemmatizer = WordNetLemmatizer()  

# Loading the intents json file, word file, classes file, and the model
intents = json.load(open('intents.json'))  
words = pickle.load(open('chatbot/words.pkl', 'rb'))  
classes = pickle.load(open('chatbot/classes.pkl', 'rb'))  
model = load_model('chatbot/chatbot_model.keras')  

# To ensure that the feedback data directory exists
if not os.path.exists('feedback'):
    os.makedirs('feedback')  # Create feedback directory if it doesn't exist

# Tokenizes and lemmatizes the inputted sentence.
def clean_sent(sentence):
    tok_sentence = nltk.word_tokenize(sentence)  # Tokenize the sentence into words
    tok_lem_sentence = [lemmatizer.lemmatize(word.lower()) for word in tok_sentence]  # Lemmatize each word and convert to lowercase
    return tok_lem_sentence

# Converts the cleaned-up sentence into a "bag-of-words" format
def bag_of_words(sentence):
    tok_lem_sentence = clean_sent(sentence)  
    bag = [0] * len(words)  # Empty bag of words
    for w in tok_lem_sentence: 
        if w in words:
            bag[words.index(w)] = 1  # Word exists then index is 1
    return np.array(bag)  # List->numpy array for using 

# Predicts the intent class from the json file for the sentence in bag-of-words format
def predict_class(sentence):
    bow = bag_of_words(sentence)  
    preds = model.predict(np.array([bow]))[0]  # Get predictions from the model
    ERROR_THRESHOLD = 0.25 

    results = [[i, r] for i, r in enumerate(preds) if r > ERROR_THRESHOLD]  # Filter predictions
    results.sort(key=lambda x: x[1], reverse=True)  # Sort predictions by probs

    intents_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results] 
    return intents_list

# Returns a random response for the predicted intent class
def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]['intent']  # Get the intent tag
        list_of_intents = intents_json['intents']  # Get list of intents from json
        for intent in list_of_intents:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])  
    return "Sorry, I cannot comprehend that at this moment."  # Default response 

# Gets data from the weather API
def get_weather_data(location, api_key, date1=None, date2=None):
    base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
    
    # Construct the URL based on the optional inputs date1 and date2
    if date1 and date2:
        url = f"{base_url}{location}/{date1}/{date2}?key={api_key}"
    elif date1:
        url = f"{base_url}{location}/{date1}?key={api_key}"
    else:
        url = f"{base_url}{location}?key={api_key}"
    
    response = requests.get(url)  # Make the request to the weather API
    
    if response.status_code == 200:  # 200 means request was successful
        return response.json()  
    else:
        return None

# Converts the weather data into a readable user-friendly form
def readable_weather(weather_data):
    if not weather_data:
        return "Unable to retrieve weather data at this time."  # Handle case where no data is available

    location = weather_data.get("resolvedAddress", "Unknown location")  # Get location 
    description = weather_data.get("description", "No description available.")  # Get description
    current_conditions = weather_data.get("currentConditions", {})  # Get current conditions
    
    current_temp = current_conditions.get("temp", "N/A")  # Get current temperature
    conditions = current_conditions.get("conditions", "N/A")  # Get current conditions
    
    response = f"Weather for {location}:\n"  
    response += f"Description: {description}\n"
    response += f"Current Temperature: {current_temp}°F\n"
    response += f"Conditions: {conditions}\n"
    
    days = weather_data.get("days", [])  # Get daily forecast data
    if days:
        response += "\nDaily Forecast:\n"
        for day in days:
            date = day.get("datetime", "N/A")  # Get date
            temp_max = day.get("tempmax", "N/A")  # Get max temp
            temp_min = day.get("tempmin", "N/A")  # Get min temp
            day_conditions = day.get("conditions", "N/A")  # Get conditions
            response += f"{date}: Highest temp is {temp_max}°F, Lowest temp is {temp_min}°F, Conditions: {day_conditions}\n"
    
    return response

# Combining all the weather data into a report form
def get_weather_report(location, date1, date2, api_key):
    response = get_weather_data(location, date1, date2, api_key)  
    if response:
        return readable_weather(response)  
    else:
        return "Unable to retrieve weather data."

# Converts the stock name to symbol to be used in the API
def get_stock_symbol(company_name):
    company_to_symbol = {
        "Apple": "AAPL",
        "Google": "GOOGL",
        "Microsoft": "MSFT",
        "Facebook": "META",
        "Amazon": "AMZN",
        "NVIDIA": "NVDA",
        "Netflix": "NFLX",
        "Adobe": "ADBE",
        "Salesforce": "CRM",
        "Cisco": "CSCO",
        "Intel": "INTC",
        "Oracle": "ORCL",
        "AMD": "AMD",
        "Texas Instruments": "TXN",
        "Qualcomm": "QCOM",
        "Broadcom": "AVGO",
        "ServiceNow": "NOW",
        "PayPal": "PYPL",
        "Intuit": "INTU",
        "Shopify": "SHOP",
        "Square": "SQ",
        "Twitter": "TWTR",
        "Uber": "UBER",
        "Zoom": "ZM",
        "Snowflake": "SNOW",
        "Palantir": "PLTR",
        "Slack": "WORK",
        "Dropbox": "DBX",
        "Cloudflare": "NET"
    }
    return company_to_symbol.get(company_name, "Unknown symbol") 

# Gets all stock price data from Alpha Vantage API
def get_stock_data(symbol, api_key, interval='5min'):
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "apikey": api_key
    }
    response = requests.get(base_url, params=params)  # Make request to stock API
    if response.status_code == 200:
        return response.json()  # Return the JSON data
    else:
        return None

# Converts stock data into a readable user-friendly form
def readable_stock(stock_data):
    if not stock_data:
        return "Unable to retrieve stock data at this time."  # Handle case where no data is available
    
    metadata = stock_data.get("Meta Data", {})  # Get metadata
    time_series = stock_data.get("Time Series (5min)", {})  # Get time series data
    
    if not metadata or not time_series:
        return "No stock data available."  # Handle case where no data is available
    
    symbol = metadata.get("2. Symbol", "Unknown symbol")  # Get stock symbol
    last_refreshed = metadata.get("3. Last Refreshed", "N/A")  # Get last refreshed time
    
    response = f"Stock data for {symbol}:\n" 
    response += f"Last Refreshed: {last_refreshed}\n"
    
    response += "\nRecent data:\n"
    for time, data in list(time_series.items())[:5]:  # Show only the latest 5 records
        response += f"{time}:\n"
        response += f"  Open: {data['1. open']}\n"
        response += f"  High: {data['2. high']}\n"
        response += f"  Low: {data['3. low']}\n"
        response += f"  Close: {data['4. close']}\n"
        response += f"  Volume: {data['5. volume']}\n"
    
    return response

# Collects user feedback and saves it to a feedbacks json file
def collect_feedback(user_input, response, feedback):
    feedback_data = {"input": user_input, "response": response, "feedback": feedback}  # Create feedback data
    with open('feedback/feedback_data.json', 'a') as f:
        json.dump(feedback_data, f)  # Save feedback to JSON file
        f.write('\n')  # New line for each entry
    print(f"Feedback collected: {feedback_data}") 

# Retrains the model with collected feedback data
def retrain_model():
    feedback_data = []
    if os.path.exists('feedback/feedback_data.json'):
        with open('feedback/feedback_data.json', 'r') as f:
            for line in f:
                feedback_data.append(json.loads(line.strip()))  # Load feedback data

    # for troubleshooting
    print(f"Total feedback entries: {len(feedback_data)}") 
    if not feedback_data:
        print("No valid training data available")
        return

    new_documents = []
    for feedback in feedback_data:
        if feedback['feedback'] == 'good':
            response_intent = None
            for intent in intents['intents']:
                if feedback['response'] in intent['responses']:
                    response_intent = intent['tag']
                    break
            if response_intent:
                new_documents.append((clean_sent(feedback['input']), response_intent))
                print(f"Added for retraining: {feedback['input']} -> {response_intent}")

    print(f"Total new documents: {len(new_documents)}")

    if not new_documents:
        print("No valid training data available for retraining.")
        return

    new_training = []
    output_empty = [0] * len(classes)  # Initialize output vector
    for document in new_documents:
        bag = []
        word_patterns = document[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

        for word in words:
            bag.append(1) if word in word_patterns else bag.append(0)  # Create bag-of-words

        output_row = list(output_empty)
        if document[1] in classes:
            output_row[classes.index(document[1])] = 1
            new_training.append([bag, output_row])

    print(f"Total new training samples: {len(new_training)}")

    if not new_training:
        print("No valid training data available for retraining.")
        return
    random.shuffle(new_training)  # Shuffle training data
    new_training = np.array(new_training, dtype=object)
    train_x = np.array([item[0] for item in new_training], dtype=np.float32)  # Features
    train_y = np.array([item[1] for item in new_training], dtype=np.float32)  # Labels

    updated_model = Sequential()  # Define a new sequential model
    updated_model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))  # Add input layer
    updated_model.add(Dropout(0.5))  # Add dropout for regularization
    updated_model.add(Dense(64, activation='relu'))  # Add hidden layer
    updated_model.add(Dropout(0.5))  # Add dropout for regularization
    updated_model.add(Dense(len(train_y[0]), activation='softmax'))  # Add output layer

    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)  # Define the optimizer
    updated_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])  # Compile the model
    updated_model.fit(train_x, train_y, epochs=100, batch_size=5, verbose=1)  # Train the model

    updated_model.save('chatbot/chatbot_model_updated.keras')  # Save the retrained model
    print("Model retrained and saved as 'chatbot/chatbot_model_updated.keras'")

def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()  # Get user input from entry box
    EntryBox.delete("0.0", END)  # Clear the entry box

    if msg != '':
        ChatBox.config(state=NORMAL)  # Enable chatbox for updating
        ChatBox.insert(END, "You: " + msg + '\n\n')  # Display user message
        ChatBox.config(foreground="#564466", font=("Helvetica", 10))  # Set font and color

        if msg.startswith("weather"):
            parts = msg.split()
            if len(parts) == 4:
                _, location, date1, date2 = parts
                api_key = "enter your api key"  # Weather API key
                weather_report = get_weather_report(location, api_key, date1, date2)  # Get weather report
                ChatBox.insert(END, "Chatbot: " + weather_report + '\n\n')  # Display weather report
            else:
                ChatBox.insert(END, "Chatbot: Please use the format 'weather [location] [date1] [date2]'\n\n")  # Invalid format message

        elif "stock price" in msg:
            company_name = msg.split(" of ")[1].replace("?", "").strip()  # Extract company name
            stock_symbol = get_stock_symbol(company_name)  # Get stock symbol
            if stock_symbol != "Unknown symbol":
                api_key = "enter your api key"  # Stock API key
                stock_data = get_stock_data(stock_symbol, api_key, '5min')  # Get stock data
                stock_report = readable_stock(stock_data)  # Convert stock data to readable format
                ChatBox.insert(END, "Chatbot: " + stock_report + '\n\n')  # Display stock report
            else:
                ChatBox.insert(END, f"Chatbot: Sorry, I don't have data for {company_name}.\n\n")  # Symbol not found message
        else:
            ints = predict_class(msg)  # Predict intent class
            res = get_response(ints, intents)  # Get response for the intent

            ChatBox.insert(END, "Chatbot: " + res + '\n\n')  # Display response

            ChatBox.config(state=DISABLED)  # Disable chatbox to prevent further editing
            ChatBox.yview(END)  # Scroll to the end of the chatbox

            feedback = input("Please provide feedback on the response (good/bad): ").strip()  # Collect feedback
            if feedback in ['good', 'bad']:
                collect_feedback(msg, res, feedback)  # Save feedback
                retrain_model()  # Retrain the model with feedback

root = Tk()  # Create main window object
root.title("Intelligent Chatbot")  # Set the title of the window
root.geometry("600x600")  # Set the window size
root.resizable(width=TRUE, height=TRUE)  # Allow resizing of the window

ChatBox = Text(root, bd=0, bg="white", height="12", width="70", font="Arial")  # Create chat box
ChatBox.config(state=DISABLED)  # Initially disable the chat box

scrollbar = Scrollbar(root, command=ChatBox.yview, cursor="heart")  # Create scrollbar
ChatBox['yscrollcommand'] = scrollbar.set  # Link scrollbar with chat box
enter_button = Button(root, font=("Helvetica", 12, 'bold'), text="Enter", width="15", height="5",  # Create enter button
                    bd=0, bg="#021ff9", activebackground="#3c9d9b", fg='#000000',
                    command=send)  # Set command to send function

EntryBox = Text(root, bd=0, bg="white", width="40", height="5", font="Helvetica")  # Create entry box

scrollbar.place(x=576, y=6, height=486)  # Place scrollbar
ChatBox.place(x=6, y=6, height=486, width=570)  # Place chat box
EntryBox.place(x=128, y=500, height=120, width=460)  # Place entry box
enter_button.place(x=6, y=500, height=120)  # Place enter button

root.mainloop()  # Start the Tkinter event loop
