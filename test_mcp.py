import os
import asyncio
from fastmcp import Client
import dotenv

async def main():
    dotenv.load_dotenv()
    async with Client(transport=os.getenv('MCP_URL')) as client:
        response = await client.call_tool('get_company_profile', {'code': '2330'})  # 台積電
        print(response)

if __name__ == "__main__":
    asyncio.run(main())