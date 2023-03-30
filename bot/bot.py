from dataclasses import dataclass
from typing import List, Optional

from langchain.agents import initialize_agent, load_tools
from langchain.llms.base import BaseLLM
from langchain.memory import ConversationBufferMemory
from langchain.tools.base import BaseTool

from .llm import llm
from .tools import (
    WandbApi,
    build_sql_db_tool,
    build_sql_table_tool,
    doc_search,
    human_in_the_loop,
    introduction,
)


@dataclass
class BotConfig:
    verbose: bool = True
    max_iterations: int = 30


@dataclass
class Bot:
    tools: List[BaseTool]
    llm: BaseLLM
    agent: str = "zero-shot-react-description"
    config: BotConfig = BotConfig()
    memory: Optional[ConversationBufferMemory] = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )

    def run(self, question: str) -> str:
        bot = initialize_agent(
            self.tools,
            self.llm,
            agent=self.agent,
            verbose=self.config.verbose,
            max_iterations=self.config.max_iterations,
            memory=self.memory,
        )
        return bot.run(question)


def get_bot(agent: str = "zero-shot-react-description") -> Bot:
    vacation_tool = build_sql_table_tool(
        tb="vacation",
        db="vacation.db",
        name="vacation_days_database",
        description="""
# Task Description:
API for vacation days. 

# Table Description:
In the table vacation, there are two columns: 
- name
- days: number of vacation days left

# Please:
- return the runnable SQL! "
- don't transform my the input.
        """,
    )
    imdb_tool = build_sql_db_tool(
        db="imdb.db",
        name="imdb_move_database",
        description="""
API for query imdb related information.

# Please:
- return the runnable SQL! "
- don't transform my the input.
        """,
    )
    wb = WandbApi()
    tools = (
        load_tools(["llm-math"], llm=llm)
        + [
            imdb_tool,
            human_in_the_loop,
            vacation_tool,
            # build_chat_in_the_loop_tool()
            introduction,
            doc_search,
        ]
        + wb.tools
    )
    return Bot(tools, llm, agent)


# TODO
# - switch bot more smoothly
