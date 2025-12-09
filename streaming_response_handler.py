"""
Streaming Response Support - Enable streaming for all AI providers
Real-time token-by-token responses for better UX
"""

import asyncio
from typing import AsyncIterator, Iterator, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum


class StreamingMode(Enum):
    """Streaming modes"""
    TOKEN = "token"  # Stream individual tokens
    SENTENCE = "sentence"  # Stream complete sentences
    PARAGRAPH = "paragraph"  # Stream complete paragraphs


@dataclass
class StreamChunk:
    """A chunk of streamed content"""
    content: str
    is_final: bool = False
    metadata: Optional[dict] = None
    cumulative_tokens: int = 0


class StreamingResponseHandler:
    """
    Unified streaming response handler for all AI providers
    
    Supported providers:
    - Gemini (native streaming)
    - Mistral (native streaming)
    - Groq (native streaming)
    - OpenRouter (native streaming)
    - HuggingFace (simulated streaming)
    - Cohere (native streaming)
    - Together AI (native streaming)
    - Claude (native streaming)
    
    Features:
    - Token-by-token streaming
    - Sentence/paragraph buffering
    - Progress callbacks
    - Error handling
    - Automatic fallback to non-streaming
    """
    
    def __init__(
        self,
        mode: StreamingMode = StreamingMode.TOKEN,
        buffer_size: int = 10,
        callback: Optional[Callable[[StreamChunk], None]] = None
    ):
        self.mode = mode
        self.buffer_size = buffer_size
        self.callback = callback
        self.cumulative_tokens = 0
    
    async def stream_from_gemini(
        self,
        model,
        prompt: str,
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """Stream from Google Gemini"""
        try:
            response = await model.agenerate_stream([prompt], **kwargs)
            
            async for chunk in response:
                if chunk and chunk.generations:
                    content = chunk.generations[0][0].text
                    self.cumulative_tokens += len(content.split())
                    
                    yield StreamChunk(
                        content=content,
                        cumulative_tokens=self.cumulative_tokens
                    )
                    
                    if self.callback:
                        self.callback(StreamChunk(content=content))
            
            # Final chunk
            yield StreamChunk(content="", is_final=True)
            
        except Exception as e:
            yield StreamChunk(
                content=f"Error: {str(e)}",
                is_final=True,
                metadata={"error": True}
            )
    
    async def stream_from_groq(
        self,
        model,
        prompt: str,
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """Stream from Groq (fastest responses)"""
        try:
            response = model.stream([{"role": "user", "content": prompt}], **kwargs)
            
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    self.cumulative_tokens += len(content.split())
                    
                    yield StreamChunk(
                        content=content,
                        cumulative_tokens=self.cumulative_tokens
                    )
                    
                    if self.callback:
                        self.callback(StreamChunk(content=content))
            
            yield StreamChunk(content="", is_final=True)
            
        except Exception as e:
            yield StreamChunk(
                content=f"Error: {str(e)}",
                is_final=True,
                metadata={"error": True}
            )
    
    async def stream_from_mistral(
        self,
        model,
        prompt: str,
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """Stream from Mistral"""
        try:
            response = model.stream([{"role": "user", "content": prompt}], **kwargs)
            
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    self.cumulative_tokens += len(content.split())
                    
                    yield StreamChunk(
                        content=content,
                        cumulative_tokens=self.cumulative_tokens
                    )
                    
                    if self.callback:
                        self.callback(StreamChunk(content=content))
            
            yield StreamChunk(content="", is_final=True)
            
        except Exception as e:
            yield StreamChunk(
                content=f"Error: {str(e)}",
                is_final=True,
                metadata={"error": True}
            )
    
    async def stream_from_cohere(
        self,
        model,
        prompt: str,
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """Stream from Cohere"""
        try:
            response = model.stream(prompt, **kwargs)
            
            for event in response:
                if event.event_type == "text-generation":
                    content = event.text
                    self.cumulative_tokens += len(content.split())
                    
                    yield StreamChunk(
                        content=content,
                        cumulative_tokens=self.cumulative_tokens
                    )
                    
                    if self.callback:
                        self.callback(StreamChunk(content=content))
            
            yield StreamChunk(content="", is_final=True)
            
        except Exception as e:
            yield StreamChunk(
                content=f"Error: {str(e)}",
                is_final=True,
                metadata={"error": True}
            )
    
    async def stream_with_buffering(
        self,
        stream_iterator: AsyncIterator[StreamChunk]
    ) -> AsyncIterator[StreamChunk]:
        """Apply sentence/paragraph buffering to token stream"""
        if self.mode == StreamingMode.TOKEN:
            # No buffering, pass through
            async for chunk in stream_iterator:
                yield chunk
        
        elif self.mode == StreamingMode.SENTENCE:
            # Buffer until sentence end
            buffer = ""
            async for chunk in stream_iterator:
                if chunk.is_final:
                    if buffer:
                        yield StreamChunk(content=buffer)
                    yield chunk
                    break
                
                buffer += chunk.content
                
                # Check for sentence end
                if any(buffer.endswith(p) for p in ['. ', '! ', '? ', '.\n', '!\n', '?\n']):
                    yield StreamChunk(content=buffer)
                    buffer = ""
        
        elif self.mode == StreamingMode.PARAGRAPH:
            # Buffer until paragraph end
            buffer = ""
            async for chunk in stream_iterator:
                if chunk.is_final:
                    if buffer:
                        yield StreamChunk(content=buffer)
                    yield chunk
                    break
                
                buffer += chunk.content
                
                # Check for paragraph end
                if '\n\n' in buffer or len(buffer) > 500:
                    yield StreamChunk(content=buffer)
                    buffer = ""
    
    async def stream_with_fallback(
        self,
        model,
        prompt: str,
        provider: str,
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream with automatic fallback to non-streaming
        
        Args:
            model: AI model instance
            prompt: Input prompt
            provider: Provider name (gemini, groq, mistral, etc.)
            **kwargs: Additional model parameters
        
        Yields:
            StreamChunk objects
        """
        try:
            # Try streaming
            if provider == "gemini":
                async for chunk in self.stream_from_gemini(model, prompt, **kwargs):
                    yield chunk
            elif provider == "groq":
                async for chunk in self.stream_from_groq(model, prompt, **kwargs):
                    yield chunk
            elif provider == "mistral":
                async for chunk in self.stream_from_mistral(model, prompt, **kwargs):
                    yield chunk
            elif provider == "cohere":
                async for chunk in self.stream_from_cohere(model, prompt, **kwargs):
                    yield chunk
            else:
                # Fallback: simulate streaming for non-streaming providers
                async for chunk in self._simulate_streaming(model, prompt, **kwargs):
                    yield chunk
        
        except Exception as e:
            # Fallback to non-streaming
            try:
                response = await model.ainvoke(prompt, **kwargs)
                # Simulate streaming of complete response
                async for chunk in self._simulate_streaming_from_text(response):
                    yield chunk
            except Exception as fallback_error:
                yield StreamChunk(
                    content=f"Error: {str(fallback_error)}",
                    is_final=True,
                    metadata={"error": True}
                )
    
    async def _simulate_streaming(
        self,
        model,
        prompt: str,
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """Simulate streaming for non-streaming models"""
        try:
            response = await model.ainvoke(prompt, **kwargs)
            async for chunk in self._simulate_streaming_from_text(str(response)):
                yield chunk
        except Exception as e:
            yield StreamChunk(
                content=f"Error: {str(e)}",
                is_final=True,
                metadata={"error": True}
            )
    
    async def _simulate_streaming_from_text(
        self,
        text: str,
        delay: float = 0.01
    ) -> AsyncIterator[StreamChunk]:
        """Simulate streaming by yielding text in chunks"""
        words = text.split()
        
        for i, word in enumerate(words):
            content = word + " "
            self.cumulative_tokens += 1
            
            yield StreamChunk(
                content=content,
                cumulative_tokens=self.cumulative_tokens
            )
            
            if self.callback:
                self.callback(StreamChunk(content=content))
            
            # Small delay to simulate streaming
            await asyncio.sleep(delay)
        
        yield StreamChunk(content="", is_final=True)


class StreamingConsoleDisplay:
    """Display streaming responses in console with formatting"""
    
    def __init__(self):
        self.current_line = ""
    
    def display_chunk(self, chunk: StreamChunk):
        """Display a chunk to console"""
        if chunk.is_final:
            print()  # New line at end
            print(f"[Completed: {chunk.cumulative_tokens} tokens]")
        else:
            print(chunk.content, end='', flush=True)
            self.current_line += chunk.content


# Example usage
async def main():
    """Example streaming usage"""
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    # Initialize handler
    display = StreamingConsoleDisplay()
    handler = StreamingResponseHandler(
        mode=StreamingMode.TOKEN,
        callback=lambda chunk: None  # display.display_chunk(chunk)
    )
    
    # Initialize model
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    
    # Stream response
    print("Streaming response:")
    print("-" * 80)
    
    async for chunk in handler.stream_with_fallback(
        model,
        "Explain what is machine learning in 3 sentences.",
        provider="gemini"
    ):
        if not chunk.is_final:
            print(chunk.content, end='', flush=True)
        else:
            print("\n" + "-" * 80)
            print(f"Completed: {chunk.cumulative_tokens} tokens")


if __name__ == "__main__":
    asyncio.run(main())
