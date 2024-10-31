from openai import OpenAI
from yaml import safe_load


def main():
    with open('config.yaml') as f:
        cfg = safe_load(f)


    client = OpenAI(api_key=cfg['open_ai'])
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke.",
            }
        ],
        model="gpt-3.5-turbo",
    )
    print(chat_completion)

if __name__ == '__main__':
    main()