import asyncio
from sqlalchemy import text
from videocuts.api.database import engine

async def add_column():
    async with engine.connect() as conn:
        print("Adding 'clip_full_transcript' column...")
        await conn.execute(text("ALTER TABLE clips ADD COLUMN clip_full_transcript TEXT;"))
        await conn.commit()
        print("Done.")

if __name__ == "__main__":
    asyncio.run(add_column())
