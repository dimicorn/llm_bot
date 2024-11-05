from openai import OpenAI
from langchain_core.vectorstores import InMemoryVectorStore


class LLM(object):
    def __init__(self, client: OpenAI, vector_store: InMemoryVectorStore) -> None:
        self.client = client
        self.vector = vector_store
    
    def generate_answer(self, context: str, message: str) -> str:
        conversation = [
        {'role': 'system', 'content':
            """You are a sales assistant at an online bank. Your task is to sell autocallables from
            the bank to potential investors and answer their questions so that they will eventually 
            want to invest their money to the bank. In your dialogue with the potential investor, 
            use the product information below. If the potential investor asks a question that is not 
            answered in the product information or previous lines of dialogue, answer 'I don't have 
            enough information to answer your question.'. Avoid mentioning facts about the company 
            or products that are not in the description given. Stick to the following scenario: if 
            the user's preferences about the type, maturity date and memory effect of product are 
            not known, find them out (in a few lines of dialogue)."""
        },
        {'role': 'user', 'content':
            f"""Autocallables info: {context}"""
        },
        {'role': 'user', 'content':
            f"""Client's preferences: {message}"""
        }
        ]
        stream = self.client.chat.completions.create(
            messages=conversation,
            model='gpt-4o',
            stream=True
        )
        answer = ''
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                answer += chunk.choices[0].delta.content
        
        return answer
    
    def search_product(self, product_desc: str, n : int = 3, pprint: bool = False) -> list[str]:        
        retriever = self.vector.as_retriever(search_type='mmr', search_kwargs={'k': n})
        docs = retriever.invoke(product_desc)
        docs = [doc.model_dump()['page_content'] for doc in docs]
        if pprint:
            print(docs)
        return docs
    
    def rag_search(self, query: str, n : int = 3, pprint: bool = False) -> str:
        results = self.search_product(query, n=n, pprint=pprint)
        context = '\n'.join(results)
        answer = self.generate_answer(context, query)
        return answer