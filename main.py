from dotenv import dotenv_values
from flask import Flask, jsonify, request
from marshmallow import Schema, fields, ValidationError
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from pinecone import Pinecone

import time

app = Flask(__name__)

queue = []
neural = []

config = dotenv_values(".env")
pc = Pinecone(api_key = config["PINECONE_API_KEY"])
index = pc.Index('secon-1')
embeddings = OpenAIEmbeddings(api_key=config["OPENAI_API_KEY"])

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

@app.route('/api/neural', methods = ['GET'])
def AddToNeural():
    # ПОМЕНЯТЬ МЕТОД НА POST!!!
    # data = request.get_json()
    # schema = RequiredSchema()
    # try:
    #     result = schema.load(data)
    # except ValidationError as e:
    #     return "Bad Request", 400

    try:
        text = "потерял ключи"
        vector = embeddings.embed_query(text)

        # vector = embeddings.embed_query(str(result['MsgText']))
        response = index.query(vector=vector, top_k=10, include_metadata=True)

        metadata = [{'id': match['id'], 'metadata': match['metadata']} for match in response['matches']]
        data = metadata[0]['metadata']['genre']

        chat = ChatOpenAI(
            temperature=0.6,
            openai_api_key=config["OPENAI_API_KEY"]
        )

        promt = "Верни мне максимально подходящие данные по объявлению, мне нужен формат json в возможном виде нескольких объектов, в json должен состоять из chat_id, msg_id и content"

        messages = [
            SystemMessage(
                content=promt
            ),
            HumanMessage(
                content=data
            ),
        ]

        content = chat.invoke(messages).content
        return jsonify({'content': content})
    except Exception as e:
        print("Analny tanets:\n", e)

if __name__ == '__main__':
    print("was started")
    app.run(host = "0.0.0.0", debug=True)