import os
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

@CrewBase
class Debate():
    """Debate crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def debater(self) -> Agent:
        return Agent(
            config=self.agents_config['debater'],
            verbose=True,
            llm=LLM(model=os.environ.get("MODEL"), api_key=os.environ.get("GROQ_API_KEY"))
        )

    @agent
    def judge(self) -> Agent:
        return Agent(
            config=self.agents_config['judge'], 
            verbose=True,
            llm=LLM(model=os.environ.get("MODEL"), api_key=os.environ.get("GROQ_API_KEY"))
        )

    @task
    def propose(self) -> Task:
        return Task(
            config=self.tasks_config['propose'],
            output_file='/output/propose.md'
        )

    @task
    def oppose(self) -> Task:
        return Task(
            config=self.tasks_config['oppose'],
            output_file='/output/oppose.md'
        )

    @task
    def decide(self) -> Task:
        return Task(
            config=self.tasks_config['decide'],
            output_file='/output/decide.md'
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
