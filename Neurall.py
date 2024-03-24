import json
import time
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from pinecone import Pinecone
from dotenv import dotenv_values

class NerualClass:
    def __init__(self, pinecone_api_key, openai_api_key):
        config = dotenv_values(".env")
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(config["INDEX"], dimension=1536)
        self.embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        self.chat = ChatOpenAI(
            temperature=0.6,
            openai_api_key=openai_api_key,
        )

    def process_data(self, text):
        vector = self.embeddings.embed_query(text)
        self.index.upsert(
            vectors=[{"id": f"{time.time()}", "values": vector, "metadata": {"genre": text}}]
        )
        return True

    def ask_question(self, text):
        vector = self.embeddings.embed_query(text)
        response = self.index.query(vector=vector, top_k=10, include_metadata=True)
        metadata = [{'id': match['id'], 'metadata': match['metadata']} for match in response['matches']]
        data = metadata[0]['metadata']['genre']
        return data

    def invoke_chat(self, text, data):
        prompt = f"Верини мне максимально подходящие данные по объявлению на запрос {text}, мне нужен формат json в возможном виде нескольких объектов, в json должен состоять из chat_id, msg_id и content, если подходящих данных нет или нет хотя бы одного из необходимых мне параметров, не возвращай мне эти данные"
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=data),
        ]
        content = self.chat.invoke(messages).content
        return content


if __name__ == "__main__":
    pinecone_api_key = "159e533b-5d20-4f3c-93c1-5277f3df9cdb"
    openai_api_key = "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"

    neural_class = NerualClass(pinecone_api_key, openai_api_key)

    # чтобы добавить данные
    text = '''{"chat_id": -1001242071007, "msg_id": 18109, "content":"Были утерены ключи район колычво примерно детсад:салнышко или ул астахова просьба позванить 89017573245"}
{"chat_id": -1001242071007, "msg_id": 18106, "content":"Найдена по адресу Коломна пр-кт Кирова 56"}
{"chat_id": -1001242071007, "msg_id": 18105, "content":"Друзья кто хочет заработать норм денег от 5.000р не чего сложного нет пишите мне в лс"}
{"chat_id": -1001242071007, "msg_id": 18104, "content":"Друзья кто хочет заработать норм денег от 5.000р не чего сложного нет пишите мне в лс"}
{"chat_id": -1001242071007, "msg_id": 18100, "content":"Найдена возле дома 56 по пр-кт Кирова у 1 подъезда"}
{"chat_id": -1001242071007, "msg_id": 18097, "content":"Нашли серьгу Весенняя 20"}'''
    neural_class.process_data(text)

    # запрос к данным
    query_text = "потерял ключи"
    result = neural_class.ask_question(query_text)
    result = neural_class.invoke_chat(query_text, result)
    print(result)
