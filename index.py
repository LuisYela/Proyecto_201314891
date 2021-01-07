import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from Util.ReadFileYela import distanciaDepartamentos
#import algoritmo

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediccion', methods=['POST'])
def prediccion():
    if request.method=='POST':
        genero = request.form['inputGenero']
        edad = request.form['inputEdad']
        anio = request.form['inputAnio']
        municipio = request.form['Select3']
        print(municipio)
        distancia = distanciaDepartamentos[municipio]
        #escalando los valores
        genero=0
        edad=(edad - 17 ) / ( 57 - 17 )
        anio=(anio - 2010 ) / ( 2019 - 2010 )
        distancia=(distancia - 6.135217051724262 ) / ( 269.7739244046257 - 6.135217051724262 )
        print(genero)
        print(edad)
        print(anio)
        print(distancia)
        return render_template('prediccion.html')

@app.route('/datos', methods=['POST'])
def crear_dato():
    return "datos recividos"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

