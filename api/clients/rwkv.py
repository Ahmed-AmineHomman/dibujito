"""
This code is freely inspired from the RWKV.cpp `inference example script <https://github.com/RWKV/rwkv.cpp/blob/master/python/inference_example.py>`__ from the `RWKV.cpp source code <https://github.com/RWKV/rwkv.cpp>`__.
The corresponding commit is the following: ``84fea22``.
The source code has an MIT licence, so we assume it can be used freely here.
"""

import os
from typing import List, Optional

from PIL import Image

from api.rwkv_cpp import RWKVModel, RWKVSharedLibrary, WorldTokenizer, sample_logits
from .base import ConversationExchange, BaseClient


class RWKVClient(BaseClient):
    """
    rwkv.cpp API client.
    """

    def __init__(
            self,
            model_path: str,
            library_path: str
    ):
        self.library = RWKVSharedLibrary(shared_library_path=library_path)
        self.model = RWKVModel(self.library, model_path)
        self.tokenizer = WorldTokenizer(
            file_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "rwkv_vocab.txt"))
        )

    def respond(
            self,
            prompt: str,
            model: Optional[str] = None,
            image_prompt: Optional[Image] = None,
            system_prompt: Optional[str] = None,
            conversation_history: Optional[List[ConversationExchange]] = None,
            max_response_size: int = 512,
            temperature: float = 0.8,
            top_p: float = 0.5,
    ) -> str:
        # build full prompt
        query = self._build_conversation(
            query=prompt,
            system_prompt=system_prompt,
            conversation_history=conversation_history
        )

        # prepare model (encode prompt & run model through it)
        tokens = self.tokenizer.encode(query)
        logits, state = self.model.eval_sequence_in_chunks(
            tokens=tokens,
            state_in=None,
            state_out=None,
            logits_out=None,
            use_numpy=True
        )

        # generate response
        response_tokens: List[int] = []
        stop_generation: bool = False
        i = 0
        token: int
        while (not stop_generation) and (i < max_response_size):
            # generate next token
            token = sample_logits(logits, temperature=temperature, top_p=top_p)

            # append tokens
            response_tokens.append(token)

            # update model, logits & states
            logits, state = self.model.eval(token, state, state, logits, use_numpy=True)

            stop_generation = token == 261
            i += 1

        # decode response tokens
        response = self._process(self.tokenizer.decode(response_tokens))

        return response

    @staticmethod
    def _build_conversation(
            query: str,
            conversation_history: Optional[List[ConversationExchange]] = None,
            system_prompt: Optional[str] = None
    ) -> str:
        """
        Builds the conversation history in RWKV-supported format.

        Notes
        -----
        More details about this format can be found in `the Hugging Face Hub model card <https://huggingface.co/BlinkDL/rwkv-6-world>`__.
        """
        output: str = ""

        # add system prompt without any introduction
        if system_prompt:
            output += RWKVClient._process(system_prompt) + "\n\n"

        # add user/assistant previous exchanges
        if conversation_history:
            for exchange in conversation_history:
                output += f"User: {RWKVClient._process(exchange.query)}\n\n"
                output += f"Assistant: {RWKVClient._process(exchange.response)}\n\n"

        # append current user query
        output += f"User: {RWKVClient._process(query)}\n\n"
        output += "Assistant:"

        return output

    @staticmethod
    def _process(text: str) -> str:
        """RWKV models do not like '\n\n' in the messages -> this method gets rid of these."""
        return text.strip().replace('\r\n', '\n').replace('\n\n', '\n')
