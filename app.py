from flask import Flask, render_template, request, jsonify
import os
from predict import predict  # Import the prediction function

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat')
def chat():
    return render_template('chatbot.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Call the prediction function
        result = predict(filepath)

        return jsonify({"status": "success", "message": f"Image uploaded successfully! Result: {result}", "prediction": result})
    return jsonify({"status": "error", "message": "No file uploaded."})

@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_message = request.json.get("message")
    response = chatbot_response(user_message)
    return jsonify({"response": response})

def chatbot_response(message):
    responses = {
        "What is oral cancer?": "Oral cancer refers to cancer that develops in the tissues of the mouth or throat.",
        "How accurate is this model?": "Our AI is trained on medical images to ensure reliable detection.",
        "Can oral cancer be treated?": "Yes, early detection improves treatment outcomes."
    }
    return responses.get(message, "I'm here to help! Ask me anything about oral cancer.")

if __name__ == '__main__':
    app.run(debug=True)
