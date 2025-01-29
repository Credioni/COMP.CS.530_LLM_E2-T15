from transformers import MarianMTModel, MarianTokenizer
from flask import Flask, request, render_template_string
# pylint: disable=all
app = Flask(__name__)


class MarianMT:
    """MarianMT Translation model"""
    def __init__(self):
        self.translation_model_name = "Helsinki-NLP/opus-mt-en-fi"
        self.translation_model     = MarianMTModel.from_pretrained(self.translation_model_name)
        self.translation_tokenizer = MarianTokenizer.from_pretrained(self.translation_model_name)

    def translate(self, text: str) -> str:
        # Tokenize the English caption for translation
        translated_inputs = self.translation_tokenizer(text, return_tensors="pt", truncation=True)

        # Use the MarianMT model to translate
        translated_outputs = self.translation_model.generate(**translated_inputs)
        translated = self.translation_tokenizer.decode(translated_outputs[0], skip_special_tokens=True)
        return translated if isinstance(translated, str) else "Error in translation"


HTML_FORM = """
<!doctype html>
<html>
    <head>
        <style>
            body {
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                font-family: Arial, sans-serif;
                background-color: #f0f0f0;
            }
            .container {
                text-align: center;
                padding: 20px;
                border: 1px solid #ccc;
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
            input[type="text"], input[type="submit"] {
                padding: 10px;
                margin: 10px;
                font-size: 16px;
            }
            h2, h3 {
                color: #333;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Enter <i>English</i> text to convert to <i>Finnish</i>:</h2>
            <form method="post">
                <input type="text" name="user_input" placeholder="Enter text here">
                <br>
                <input type="submit" value="Convert">
            </form>
            <h3>Converted Text:</h3>
            {% if result %}
                <h3>{{ result }}</h3>
            {% endif %}
        </div>
    </body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        user_input = request.form["user_input"]
        result = marian.translate(user_input)
    return render_template_string(HTML_FORM, result=result)

if __name__ == "__main__":
    marian = MarianMT()
    app.run(debug=True)
