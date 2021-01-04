import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediccion')
def prediccion():
    return render_template('prediccion.html')

@app.route('/datos', methods=['POST'])
def crear_dato():
    return "datos recividos"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

