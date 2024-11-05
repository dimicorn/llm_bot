from openai import OpenAI
from yaml import safe_load
from telebot.async_telebot import AsyncTeleBot
from asyncio import run
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llm import LLM


with open('config.yaml') as f:
    cfg = safe_load(f)

document_loader = PyPDFDirectoryLoader(cfg['data_path'])
textsplitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
raw_data = document_loader.load_and_split(textsplitter)
embeddings = OpenAIEmbeddings()
vector_store = InMemoryVectorStore.from_documents(raw_data, embeddings)

bot = AsyncTeleBot(cfg['tg_bot'])

@bot.message_handler(commands=['start'])
async def send_welcome(message):
    await bot.send_message(message.chat.id, 'Hi!')

@bot.message_handler()
async def send_text(message):
    client = OpenAI(api_key=cfg['open_ai'])
    llm = LLM(client, vector_store)
    answer = llm.rag_search(message.text)
    await bot.send_message(message.chat.id, answer)

def main():
    run(bot.polling(none_stop=True))


if __name__ == '__main__':
    main()