"""
Conversational AI system

Wraps language model with:
- Conversation context
- Memory retrieval
- System instructions
- Response generation
"""

from typing import Optional, List, Dict
from src.inference.generator import TextGenerator, GenerationConfig
from src.memory.memory import MemoryStore, ConversationMemory


class ConversationalAI:
    """Conversational AI with memory and context management."""

    def __init__(
        self,
        model,
        tokenizer,
        memory_store: Optional[MemoryStore] = None,
        device: str = "cpu",
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize conversational AI.

        Args:
            model: Language model
            tokenizer: Tokenizer
            memory_store: Memory store for conversation history
            device: Device for inference
            system_instruction: System prompt/instruction
        """
        self.generator = TextGenerator(model, tokenizer, device)
        self.tokenizer = tokenizer

        # Initialize memory
        if memory_store is None:
            self.memory_store = MemoryStore()
        else:
            self.memory_store = memory_store

        self.conversation_memory = ConversationMemory(self.memory_store)
        self.system_instruction = system_instruction or (
            "You are Voxline, a helpful AI assistant. "
            "Be concise, accurate, and respectful."
        )

    def chat(
        self,
        user_message: str,
        include_memory: bool = True,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: Optional[float] = 0.9,
    ) -> str:
        """
        Process user message and generate response.

        Args:
            user_message: User input
            include_memory: Whether to include conversation history
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Assistant response
        """
        # Add user message to memory
        self.conversation_memory.add_message("user", user_message)

        # Build prompt with context
        if include_memory:
            context = self.conversation_memory.get_context(num_messages=5)
            prompt = f"{self.system_instruction}\n\nConversation:\n{context}\n\nAssistant:"
        else:
            prompt = f"{self.system_instruction}\n\nUser: {user_message}\n\nAssistant:"

        # Generate response
        config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )

        response = self.generator.generate(prompt, config, return_text=True)

        # Clean response
        response = self._clean_response(response, prompt)

        # Add to memory
        self.conversation_memory.add_message("assistant", response)

        return response

    def multi_turn_chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 100,
    ) -> List[str]:
        """
        Process multiple messages.

        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}
            max_new_tokens: Max tokens per response

        Returns:
            List of assistant responses
        """
        responses = []

        for msg in messages:
            if msg["role"] == "user":
                response = self.chat(
                    msg["content"],
                    max_new_tokens=max_new_tokens,
                )
                responses.append(response)
            elif msg["role"] == "assistant":
                # Add assistant message to memory without generating
                self.conversation_memory.add_message("assistant", msg["content"])

        return responses

    def get_context(self, num_messages: int = 5) -> str:
        """Get conversation context."""
        return self.conversation_memory.get_context(num_messages=num_messages)

    def clear_conversation(self):
        """Clear current conversation."""
        self.conversation_memory.clear()

    def search_memory(self, query: str, limit: int = 5) -> List[Dict]:
        """Search long-term memory."""
        memories = self.memory_store.search_memories(query, limit=limit)
        return [m.to_dict() for m in memories]

    def add_memory(
        self,
        content: str,
        memory_type: str = "semantic",
        tags: Optional[List[str]] = None,
    ) -> str:
        """Add custom memory entry."""
        return self.memory_store.add_memory(
            content=content,
            memory_type=memory_type,
            tags=tags or [],
        )

    def export_conversation(self, path: str):
        """Export conversation to file."""
        self.conversation_memory.export(path)

    def _clean_response(self, response: str, prompt: str) -> str:
        """
        Clean generated response.

        Args:
            response: Generated text
            prompt: Original prompt

        Returns:
            Cleaned response
        """
        # Remove prompt from beginning if present
        if response.startswith(prompt):
            response = response[len(prompt) :].strip()

        # Remove common artifacts
        response = response.split("\nUser:")[0].strip()
        response = response.split("\n\n")[0].strip()

        return response

    def set_system_instruction(self, instruction: str):
        """Set system instruction."""
        self.system_instruction = instruction

    def get_memory_store(self) -> MemoryStore:
        """Get underlying memory store."""
        return self.memory_store
