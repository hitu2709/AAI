from langchain_core.callbacks import BaseCallbackHandler


class LLMCallCounter(BaseCallbackHandler):

    def __init__(self):
        self.calls = 0

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.calls += 1

    def reset(self):
        self.calls = 0