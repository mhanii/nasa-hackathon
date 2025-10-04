import os
import json
import asyncio
from dotenv import load_dotenv
from typing import Any, Optional, Union, List
import google.generativeai as genai
from neo4j_graphrag.embeddings import Embedder
from neo4j_graphrag.llm.base import LLMInterface
from neo4j_graphrag.llm.types import LLMResponse
from neo4j_graphrag.types import LLMMessage
from neo4j_graphrag.message_history import MessageHistory
from google.genai import types

class GeminiLLM(LLMInterface):
    """Custom Gemini LLM implementation using google-generativeai package"""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash", # Updated to a common model name
        api_key: Optional[str] = None,
        model_params: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(model_name=model_name, model_params=model_params, **kwargs)

        # Configure Gemini API
        api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set.")
        genai.configure(api_key=api_key)

        # Initialize model
        self.model = genai.GenerativeModel(model_name)

        # Default generation config
        self.generation_config = {
            "temperature": model_params.get("temperature", 0) if model_params else 0,
            "top_p": model_params.get("top_p", 0.95) if model_params else 0.95,
            "top_k": model_params.get("top_k", 40) if model_params else 40,
            "max_output_tokens": model_params.get("max_output_tokens", 8192) if model_params else 8192,
        }

    # def _format_messages(self, messages: List[LLMMessage]) -> list:
    #     """
    #     Converts a list of LLMMessage objects to the Gemini API's expected format.
    #     Maps 'system' role to 'user' and ensures correct structure.
    #     """
    #     gemini_messages = []
    #     for msg in messages:
    #         # Gemini API uses 'user' and 'model' for turn-based conversation.
    #         # The 'system' role is mapped to 'user'.
    #         role = "user" if msg.role in ["user", "system"] else "model"
    #         gemini_messages.append({
    #             "role": role,
    #             "parts": [msg.content]
    #         })
    #     return gemini_messages

    def invoke(
        self,
        input: Union[str, List[LLMMessage]],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """Send input to Gemini and return response."""

        all_messages: List[LLMMessage] = []

        # 1. Add system instruction
        if system_instruction:
            all_messages.append(LLMMessage(role="system", content=system_instruction))

        # 2. Add message history
        if message_history:
            history_list = message_history.messages if isinstance(message_history, MessageHistory) else message_history
            all_messages.extend(history_list)

        # 3. Add the main input
        if isinstance(input, list):
            # Handles the case from the traceback where the full conversation is passed as `input`
            all_messages.extend(input)
        elif isinstance(input, str):
            # Handles the standard case where `input` is the final user message
            all_messages.append(LLMMessage(role="user", content=input))
        
        # 4. Format all messages for the Gemini API
        gemini_formatted_messages = self._format_messages(all_messages)

        try:
            # Use generate_content with the complete, formatted conversation history
            response = self.model.generate_content(
                gemini_formatted_messages,
                generation_config=self.generation_config
            )
            
            content = response.text
            return LLMResponse(content=content)

        except Exception as e:
            print(f"Gemini API error: {str(e)}")

    async def ainvoke(
        self,
        input: Union[str, List[LLMMessage]],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """Async version - runs in executor since google-generativeai doesn't have native async"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.invoke,
            input,
            message_history,
            system_instruction
        )

class GeminiEmbeddings(Embedder):
    """Custom Gemini Embeddings using google-generativeai package"""
    
    def __init__(
        self,
        model: str = "gemini-embedding-001",
        api_key: Optional[str] = None,
    ):
        self.model = model
        
        # Configure API
        api_key = api_key or os.getenv('GOOGLE_API_KEY')
        genai.configure(api_key=api_key)
    
    def embed_query(self, text: str) -> list[float]:
        """Generate embeddings for a single text"""
        result = genai.embed_content(
            model=self.model,
            content=text,
            task_type="retrieval_query",
            output_dimensionality=768

        )
        return result['embedding']
    
    async def aembed_query(self, text: str) -> list[float]:
        """Async version of embed_query"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_query, text)
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts"""
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document",
                output_dimensionality=768

            )
            embeddings.append(result['embedding'])
        return embeddings
    
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Async version of embed_documents"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_documents, texts)


# Example usage
if __name__ == "__main__":

    load_dotenv('.env', override=True)

    # Test Gemini LLM
    llm = GeminiLLM(
        model_name="gemini-2.5-flash",
        model_params={"temperature": 0}
    )
    
    response = llm.invoke("What is a knowledge graph?")
    print("LLM Response:", response.content)
    
    # Test Gemini Embeddings
    embedder = GeminiEmbeddings()
    
    embedding = embedder.embed_query("Hello world")
    print(f"\nEmbedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")