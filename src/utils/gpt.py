# The script contains helper functions and classes that query openAI api to generate text using GPT-3.5 and GPT-4 model.

from abc import ABC, abstractmethod
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed
import ollama

MAX_ATTEMPTS = 3
WAIT_TIME = 10

class Wrapper(ABC):
    @retry(stop=stop_after_attempt(MAX_ATTEMPTS), wait=wait_fixed(WAIT_TIME))
    def summarize(self, text, summary_token_size = 200):
        return self._summarize(text, summary_token_size)

    @retry(stop=stop_after_attempt(MAX_ATTEMPTS), wait=wait_fixed(WAIT_TIME))
    def classify_sentiment(self, text):
        return self._classify_sentiment(text)

    @abstractmethod
    def _summarize(self, prompt: str, max_tokens: int) -> str:
        pass

    @abstractmethod
    def _classify_sentiment(self, prompt: str) -> str:
        pass


class GPT(Wrapper):
    def __init__(self, api_key, model_version):
        self.model_version = model_version
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)

    def _summarize(self, content: str, max_tokens: int) -> str:
        prompt = f"""Summarize the following news article within {max_tokens} tokens:\n{content}\nSummary:"""
        response = self.client.chat.completions.create(
            model=self.model_version,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
        )
        summary = response.choices[0].message.content
        return summary

    def _classify_sentiment(self, content: str) -> str:
        prompt = f"""Classify the sentiment of the following movie reviews and output the sentiment as a single word "positive" or "negative". Text:\n{content}"""
        response = self.client.chat.completions.create(
            model=self.model_version,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
        )
        sentiment = response.choices[0].message.content
        return sentiment
    

def summarize_text_llama3(model, text, max_tokens=200):

    response = ollama.chat(model=model, messages=[
        {
            'role': 'system',
            'content': 'You are a helpful assistant.'
        },
        {
            'role': 'user',
            'content': f'Summarize the following news article within {max_tokens} tokens:\n {text}.\nSummary:',
        }
    ])
    result = response['message']['content']
    return result


def classify_sentiment_llama3(model, text):

    response = ollama.chat(model=model, messages=[
        {
            'role': 'system',
            'content': 'You are a helpful assistant.'
        },
        {
            'role': 'user',
            'content': f'Classify the sentiment of the following movie reviews and output the sentiment as a single word "positive" or "negative". Text:\n{text}',
        }
    ])
    result = response['message']['content']
    return result