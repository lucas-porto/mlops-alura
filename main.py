from flask import Flask, request, jsonify
import pickle
import os
from flask_basicauth import BasicAuth
from textblob import TextBlob
from googletrans import Translator
from sklearn.linear_model import LinearRegression


#usada para realizar o tratamento do json
colunas = ['tamanho', 'ano', 'garagem'] 

#carregando o modelo
modelo = pickle.load(open('models/modelo.sav', 'rb'))

#inicializando o objeto flask
app = Flask(__name__)

#configurando a autenticação básica
app.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME')
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD')

basic_auth = BasicAuth(app)

#instanciando o tradutor
translator = Translator()

#criando os end-points
@app.route('/')
def home():
    return "Minha primeira API."

#via url
@app.route('/sentimento/<frase>')
@basic_auth.required
def sentimento(frase):
    frase_en = translator.translate(frase, dest='en')
    tb_en = TextBlob(frase_en.text)
    polaridade = tb_en.sentiment.polarity
    return "polaridade: {}".format(polaridade)

#via post
@app.route('/cotacao/', methods=['POST'])
@basic_auth.required
def cotacao():
    data = request.get_json()
    dados_input = [data[col] for col in colunas]
    preco = modelo.predict([dados_input])
    return jsonify(preco = preco[0])

#garantia de que o código só execute quando for executado via
#iniciar nosso programa agora: gunicorn -b :$PORT main:app
#basicamente seria chamar o python main.py no bash
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')