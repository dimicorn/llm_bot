from tomllib import load
from telebot.async_telebot import AsyncTeleBot
from asyncio import run
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llm import LLM
import streamlit as st
# import os

# Everything is accessible via the st.secrets dict:

# st.write("DB username:", st.secrets["db_username"])
# st.write("DB password:", st.secrets["db_password"])

# print(st.secrets)
# with open(f'{os.path.expanduser('~')}/.streamlit/config.toml', 'rb') as f:
#     cfg = load(f)

document_loader = PyPDFDirectoryLoader(st.secrets['data_path'])
textsplitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
raw_data = document_loader.load_and_split(textsplitter)
embeddings = OpenAIEmbeddings()
vector_store = InMemoryVectorStore.from_documents(raw_data, embeddings)

bot = AsyncTeleBot(st.secrets['tg_bot'])

@bot.message_handler(commands=['start'])
async def send_welcome(message):
    await bot.send_message(message.chat.id, 'Hi!')

@bot.message_handler()
async def send_text(message):
    client = OpenAI(api_key=st.secrets['open_ai'])
    llm = LLM(client, vector_store)
    answer = llm.rag_search(message.text, pprint=True)
    await bot.send_message(message.chat.id, answer)

def main():
    run(bot.polling(none_stop=True))


if __name__ == '__main__':
    main()