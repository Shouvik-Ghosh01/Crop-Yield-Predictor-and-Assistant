from flask import Flask, render_template, request, jsonify
import pickle
import requests
from transformers import AutoTokenizer, AutoConfig, T5ForConditionalGeneration
import safetensors.torch  # Ensure safetensors is installed
import torch

app = Flask(__name__)

# Load ML models
dtr = pickle.load(open("models/dtr_2.pkl", "rb"))
preprocessor = pickle.load(open("models/preprocessor_2.pkl", "rb"))

# OpenWeather API key
user_api = "a80014a6052fb80f913dee1e7459346f"

# Load the Flan-T5 model configuration and tokenizer
model_path = "flan_t5_finetuned_2"
config = AutoConfig.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")  # Use compatible tokenizer

# Initialize the model with the configuration
model = T5ForConditionalGeneration(config)

# Load the model weights from model.safetensors with strict=False
safetensors_path = f"{model_path}/model.safetensors"
state_dict = safetensors.torch.load_file(safetensors_path)
model.load_state_dict(state_dict, strict=False)  # Allow partial loading

def get_temperature(state_name):
    """Fetches current temperature of the state using OpenWeather API."""
    complete_api_link = f"https://api.openweathermap.org/data/2.5/weather?q={state_name}&appid={user_api}"
    api_link = requests.get(complete_api_link)
    api_data = api_link.json()

    if api_data['cod'] == '404':
        return None  # Return None if state not found
    else:
        temp_city = api_data['main']['temp'] - 273.15
        return round(temp_city, 2)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_temperature', methods=['POST'])
def fetch_temperature():
    data = request.get_json()
    state = data.get('state')
    temp = get_temperature(state)

    if temp is None:
        return jsonify({"error": "Invalid state name. Please check the state name."}), 400

    return jsonify({'temperature': temp})

@app.route('/predict_yield', methods=['POST'])
def predict_yield():
    data = request.form
    crop = data.get('crop')
    season = data.get('season')
    state = data.get('state')
    area = float(data.get('area'))
    rainfall = float(data.get('annual_rainfall'))

    temp = get_temperature(state)
    if temp is None:
        return jsonify({"error": "Invalid state name. Please check the state name."}), 400

    raw_features = [[crop, season, state, area, rainfall, temp]]
    processed_features = preprocessor.transform(raw_features)
    prediction = dtr.predict(processed_features)[0]

    return jsonify({'yield_prediction': prediction})

@app.route('/ask_question', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question')

    # Encode question and generate answer
    inputs = tokenizer("question: " + question, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'])
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)