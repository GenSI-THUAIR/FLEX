import asyncio
import time
from openai import AsyncOpenAI
import os

class RateLimitedLLMClient:
    def __init__(self, max_concurrent=3, rpm_limit=45):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.rpm_limit = rpm_limit
        self.request_times = []
        self.client = AsyncOpenAI(
            api_key=os.environ['API_KEY'],
            base_url=os.environ['BASE_URL']
        )
    
    async def chat_completion(self, model, messages, temperature=0, stream=False, max_tokens=32768):
        async with self.semaphore:
            await self._check_rate_limit()
            
            try:
                if stream:
                    # Streaming mode
                    response = await self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        stream=True,
                        max_tokens=max_tokens
                    )
                    
                    content = ""
                    async for chunk in response:
                        if hasattr(chunk, 'choices') and chunk.choices and len(chunk.choices) > 0:
                            if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                                if chunk.choices[0].delta.content is not None:
                                    content += chunk.choices[0].delta.content
                    
                    return content.strip()
                else:
                    # Non-streaming mode
                    response = await self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        stream=False
                    )
                    return response.choices[0].message.content.strip()
            
            except Exception as e:
                if "429" in str(e):
                    await asyncio.sleep(60)
                    return await self.chat_completion(model, messages, temperature, stream)
                raise e
    
    async def _check_rate_limit(self):
        now = time.time()
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        if len(self.request_times) >= self.rpm_limit:
            sleep_time = 60 - (now - self.request_times[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        self.request_times.append(now)
