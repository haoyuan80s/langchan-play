from dataclasses import dataclass
from typing import List

from langchain.agents import initialize_agent, load_tools
from langchain.llms.base import BaseLLM
from langchain.tools.base import BaseTool

from .llm import llm
from .tools import (
    WandbApi,
    build_human_in_the_loop_tool,
    build_sql_db_tool,
    build_sql_table_tool,
)


@dataclass
class AgentConfig:
    verbose: bool = True
    max_iterations: int = 10


@dataclass
class Agent:
    tools: List[BaseTool]
    llm: BaseLLM
    agent_name: str = "zero-shot-react-description"
    config: AgentConfig = AgentConfig()

    def run(self, question: str) -> None:
        bot = initialize_agent(
            self.tools,
            self.llm,
            agent=self.agent_name,
            verbose=self.config.verbose,
            max_iterations=self.config.max_iterations,
        )
        bot.run(question)


def get_vaction_agent():
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
    human_in_the_loop_tool = build_human_in_the_loop_tool()
    tools = load_tools(["llm-math"], llm=llm) + [
        vacation_tool,
        human_in_the_loop_tool,
    ]

    return Agent(tools, llm)


def get_imdb_agent():
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
    # human_in_the_loop_tool = build_human_in_the_loop_tool()
    tools = load_tools(["llm-math"], llm=llm) + [
        imdb_tool,
        # human_in_the_loop_tool,
    ]

    return Agent(tools, llm)


def get_combined_agent():
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
    human_in_the_loop_tool = build_human_in_the_loop_tool()
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
            human_in_the_loop_tool,
            vacation_tool,
        ]
        + wb.tools
    )
    return Agent(tools, llm)


# TODO join Agents
