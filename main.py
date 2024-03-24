from dotenv import dotenv_values
from flask import Flask, jsonify, request
from marshmallow import Schema, fields, ValidationError
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from pinecone import Pinecone
from Neurall import NerualClass

import time

app = Flask(__name__)

queue = []
neural = []

config = dotenv_values(".env")
pc = Pinecone(api_key = config["PINECONE_API_KEY"])
index = pc.Index('secon-1')
embeddings = OpenAIEmbeddings(api_key=config["OPENAI_API_KEY"])
neuralObj = NerualClass(pinecone_api_key=config["PINECONE_API_KEY"], openai_api_key=config["OPENAI_API_KEY"])

class RequiredSchema(Schema):
    ChatId = fields.String(required=True)
    MsgId = fields.String(required=True)
    MsgText = fields.String(required=True)

@app.route('/api/ping', methods = ['GET'])
def Ping():
    print('pong')
    return jsonify({'message': 'pong'})

"""
Queue routes
"""
@app.route('/api/queue', methods = ['GET'])
def GetQueue():
    return jsonify({'queue': queue})

@app.route('/api/queue', methods = ['POST'])
def AddToQueue():
    data = request.get_json()
    schema = RequiredSchema()
    try:
        result = schema.load(data)
    except ValidationError as e:
        return "Bad Request", 400

    queue.append((result['UserId'], result['MessageText']))
    return jsonify({'total_entities_neural': f'{len(queue)}'})

"""
Neural Routes
"""
# @app.route('/api/neural', methods = ['GET'])
# def GetNeural():
#     return jsonify({'neural': neural})

@app.route('/api/neural', methods = ['POST'])
def AddToNeural():
    data = request.get_json()
    schema = RequiredSchema()
    try:
        result = schema.load(data)
    except ValidationError as e:
        return "Bad Request", 400

    # text = str(result)
    # text = text.replace("'", '"')
    # print(text)
    neuralObj.process_data(result['MsgText'])
    return jsonify({'content': 'data was added'})

@app.route('/api/neural', methods = ['GET'])
def GetFromNeural():
    data = request.get_json()
    schema = RequiredSchema()
    try:
        result = schema.load(data)
    except ValidationError as e:
        return "Bad Request", 400

    text = result['MsgText']
    result = neuralObj.ask_question(text)
    result = neuralObj.invoke_chat(text, result)
    return jsonify({'content': result})

if __name__ == '__main__':
    print("was started")
    app.run(host = "0.0.0.0", debug=True)