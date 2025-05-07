# %%
import uuid
import psycopg
import asyncio
from langchain_postgres import PostgresChatMessageHistory

class ChatHistory:
    def __init__(self, conn_info, table_name):
        self.conn_info = conn_info
        self.table_name = table_name

    def create_tables(self, table_name):
        with psycopg.connect(self.conn_info) as sync_conn:
            PostgresChatMessageHistory.create_tables(sync_conn, table_name)

    async def acreate_tables(self, table_name):
        async with await psycopg.AsyncConnection.connect(
            self.conn_info
        ) as async_conn:
            PostgresChatMessageHistory.acreate_tables(async_conn, table_name)

    async def get_history(self, session_id):
        async with await psycopg.AsyncConnection.connect(
            self.conn_info
        ) as async_conn:
            history = PostgresChatMessageHistory(
                self.table_name, session_id, async_connection=async_conn
            )
            return history

    async def aadd_messages(self, session_id, messages):
        async with await psycopg.AsyncConnection.connect(
            self.conn_info
        ) as async_conn:
            history = PostgresChatMessageHistory(
                self.table_name, session_id, async_connection=async_conn
            )
            await history.aadd_messages(messages)
    
    def __call__(self, session_id):
        return self.get_history(session_id)


# %%
