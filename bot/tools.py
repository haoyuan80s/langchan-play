import sqlite3
from dataclasses import dataclass
from functools import cached_property
from typing import Dict, List

import numpy as np
import pandas as pd
import wandb
from langchain import PromptTemplate
from langchain.agents import tool
from langchain.chains import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseLanguageModel
from langchain.tools.base import BaseTool

from .llm import llm

DATA_PATH = "/Users/bytedance/Projects/qa-chain/langchan-play/bot/data/"


def tb_sql_prompt(db_name, tb_name):
    """Automatically generate a SQL prompt for the given LLM and prompt template."""
    con = sqlite3.connect(DATA_PATH + db_name)
    c = con.cursor()
    try:
        table_info = list(c.execute(f"PRAGMA table_info({tb_name});"))
        return f"""
            This is a summary of the table {tb_name} in database {db_name}:
            table info: {table_info}
            ----------------------------
            I want you to act as a sqlite3 SQL engine, and return the result to be executed by a real SQL engine.
            """
    finally:
        con.close()


def db_sql_prompt(db_name):
    """Automatically generate a SQL prompt for the given LLM and prompt template."""

    con = sqlite3.connect(DATA_PATH + db_name)
    c = con.cursor()
    tables = list(c.execute("SELECT name FROM sqlite_master WHERE type='table';"))
    table_names = [table[0] for table in tables]
    table_info = {}
    for table in table_names:
        table_info[table] = list(c.execute(f"PRAGMA table_info({table});"))
    con.close()
    return f"""
This is a summary of the database {db_name}:
table names: {table_names}
table info: {list(table_info.items())}
----------------------------
I want you to act as a sqlite3 SQL engine, and return the result to be executed by a real SQL engine.
        """


@dataclass
class SqlTable:
    lang: str
    name: str
    tabel_schema: Dict[str, str]
    description: str
    llm: BaseLanguageModel
    tool_description: str

    @property
    def prompt(self) -> str:
        return f"""
I want you to act as a {self.lang} SQL engine.

table name: {self.name}
description: {self.description}
schema: {', '.join([f'{k} {v}' for k, v in self.tabel_schema.items()])}
        """


def build_sql_table_tool(tb, db, name, description) -> BaseTool:
    def fn(sql_task_description):
        con = sqlite3.connect(DATA_PATH + db)
        cur = con.cursor()
        template = f"""
        {tb_sql_prompt(db, tb)}
        SQL task: {{sql_task_description}}.
        Please select specific columns and rows. (not select *)
        """

        prompt = PromptTemplate(
            input_variables=["sql_task_description"],
            template=template,
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        sql = chain.run(sql_task_description)
        try:
            res = str(list(cur.execute(sql)))
        except Exception as e:
            return f"""Sorry I can't run the SQL: {sql}. Error: {e}"""
        finally:
            con.close()
        if not res:
            return f"""I ran the SQL: {sql}. But I got an empty result."""
        else:
            return f"""I ran the SQL: {sql}. And I got the result: {res}"""

    fn.__name__ = name
    fn.__doc__ = description
    return tool(fn)  # type: ignore


def build_sql_db_tool(db, name, description) -> BaseTool:
    def fn(sql_task_description):
        con = sqlite3.connect(DATA_PATH + db)
        cur = con.cursor()
        template = f"""
        {db_sql_prompt(db)}
        SQL task: {{sql_task_description}}.
        SQL row limit: 100
        Please select specific columns and rows. (not select *)
        """

        prompt = PromptTemplate(
            input_variables=["sql_task_description"],
            template=template,
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        sql = chain.run(sql_task_description)
        try:
            res = str(list(cur.execute(sql)))
        except Exception as e:
            return f"""Sorry I can't run the SQL: {sql}. Error: {e}"""
        finally:
            con.close()
        if not res:
            return f"""I ran the SQL: {sql}. But I got an empty result."""
        else:
            return f"""I ran the SQL: {sql}. And I got the result: {res}"""

    fn.__name__ = name
    fn.__doc__ = description
    return tool(fn)  # type: ignore


def build_agent_b_in_the_loop_tool():
    template = """Assistant is a large language model trained by OpenAI. 
    - Give short message (<20 words)

    {history}
    Human: {human_input}
    Assistant:"""

    prompt = PromptTemplate(
        input_variables=["history", "human_input"], template=template
    )

    chatgpt_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=ConversationBufferWindowMemory(k=5),
    )

    @tool
    def chat_in_the_loop(query):
        """A chat bot in the loop tool. Use this for chatting.

        Example queries: Hello, How are you?, What is your name?, etc.
        """

        return chatgpt_chain.predict(human_input=query)

    return chat_in_the_loop


@tool
def human_in_the_loop(query):
    """Human in the loop tool. Use this if you are not sure what to do."""
    ans = input(
        f"""
--------------------------------------------------------
Hi human, can you help solve the answer: "{query}"?\n"
Answer: 
"""
    )
    print("--------------------------------------------------------")
    return ans


@tool
def introduction(_):
    """Introduction of the assistant."""
    return """
Hi, I am an assistant. I am based on a large language model trained by OpenAI.
I am also familar with company specific tools and databases."""


@tool
def doc_search(query):
    """A tool for search information in doc."""

    with open(DATA_PATH + "doc.txt", "r") as f:
        lines = f.readlines()

    embeddings = OpenAIEmbeddings()
    keys = np.array([embeddings.embed_query(line) for line in lines])
    query = np.array(embeddings.embed_query(query))
    # find id of key that is closest to query, measured by cosine distance
    id = np.argmin(np.sum((keys - query) ** 2, axis=1))
    return lines[id]


@dataclass
class WandbApi:
    project: str = "my-awesome-project"
    entity: str = "haoyuan"

    @property
    def runs(self):
        api = wandb.Api()
        runs = api.runs(self.entity + "/" + self.project)
        return runs

    @property
    def dataframe(self):
        losses, accs, config_list, name_list = [], [], [], []
        for run in self.runs:
            losses.append(run.summary._json_dict["loss"])
            accs.append(run.summary._json_dict["acc"])

            config_list.append(
                {k: v for k, v in run.config.items() if not k.startswith("_")}
            )

            name_list.append(run.name)

        runs_df = pd.DataFrame(
            {"loss": losses, "acc": accs, "config": config_list, "name": name_list}
        )
        return runs_df

    @cached_property
    def name_to_id(self):
        return {run.name: run.id for run in self.runs}

    @property
    def tools(self) -> List[BaseTool]:
        @tool
        def get_model_loss(name):
            """Get the loss of a model.
            Name is a string consisting  only:
                - letters
                - numbers
                - dash
            """
            name = name.strip(r"'")
            try:
                return str(
                    self.dataframe[self.dataframe["name"] == name].iloc[0]["loss"]
                )
            except Exception as e:
                return f"Sorry, I can't find the loss of {name}. Error: {e}"

        @tool
        def get_all_model_names(_):
            """Get the all model names"""
            return str(self.dataframe["name"].tolist())

        @tool
        def plot_model_training_loss_curve(name):
            """Plot the training loss curve of all models. And open a learning curve tab in your browser."""

            name = name.strip(r"'")
            import webbrowser

            try:
                id = self.name_to_id[name]
                url = f"https://wandb.ai/haoyuan/my-awesome-project/runs/{id}?workspace=user-haoyuan"
                webbrowser.open(url)
                return f"I opened the training loss curve of {name} in your browser, with url: {url}"
            except Exception as e:
                return f"Sorry, I can't find the loss of {name}. Error: {e}"

        @tool
        def get_model_config(name):
            """Get the config of a model.

            Name is a string consisting  only:
                - letters
                - numbers
                - dash
            """
            name = name.strip(r"'")
            try:
                return str(
                    self.dataframe[self.dataframe["name"] == name].iloc[0]["config"]
                )
            except Exception as e:
                return f"Sorry, I can't find the config of {name}. Error: {e}"

        return [
            get_model_loss,
            get_all_model_names,
            get_model_config,
            plot_model_training_loss_curve,
        ]  # type: ignore
