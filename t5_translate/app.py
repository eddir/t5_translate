# Import the Flask library
from flask import Flask, jsonify, request

from transformers import T5ForConditionalGeneration, T5Tokenizer

# Create an instance of the Flask class. This is your WSGI application.
app = Flask(__name__)


# Define a route for the root URL ("/")
@app.route('/')
def hello_world():
    return 'Welcome to my simple Flask REST API!'


# Define a route for the /translate URL with a POST method to translate text from English to Russian
@app.route('/translate', methods=['POST'])
def translate():
    if not request.json or 'text' not in request.json:
        return jsonify({'error': 'the request must be in JSON with a text key'}), 400

    prefix = 'translate to ru: '
    src_text = prefix + request.json['text']

    # translate Russian to Chinese
    input_ids = tokenizer(src_text, return_tensors="pt")

    generated_tokens = model.generate(**input_ids.to(device))

    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    return jsonify({'translation': result[0]})


# Check if the executed file is the main program and run the app
if __name__ == '__main__':
    device = 'cpu'  # or 'cpu' for translate on cpu

    model_name = 'utrobinmv/t5_translate_en_ru_zh_large_1024'
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    app.run(debug=True)
