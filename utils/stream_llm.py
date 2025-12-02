import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

async def stream_llm(messages):
    client = AsyncOpenAI( api_key=os.environ.get("API_KEY"),
        base_url=os.environ.get("BASE_URL"))
    
    stream = await client.chat.completions.create(
        model=os.environ.get("MODEL"),
        messages=messages,
        stream=True,
        temperature=0.7
    )
    
    async for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content

if __name__ == "__main__":
    import asyncio
    
    async def test():
        messages = [{"role": "user", "content": "Hello!"}]
        async for chunk in stream_llm(messages):
            print(chunk, end="", flush=True)
        print()
    
    asyncio.run(test()) 