import re
import asyncio
from metagpt.actions import Action
from metagpt.config2 import config
from metagpt.const import METAGPT_ROOT
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.tools import Tool

class WriteCode(Action):
    async def run(self, instruction: str) -> str:
        prompt = f"""
        # Dependent upon metagpt.tools.natural_language_write_file
        You are a Python programmer agent. Your task is to write Python code based on the user's instruction.
        Ensure the code is clean, well-commented, and follows best practices.
        Use the available tools like `natural_language_write_file` to save the code to a file if needed.

        Instruction: {instruction}

        Return only the Python code block, enclosed in
```
python...
```
"""
        rsp = await self._aask(prompt)
        # Extract code from the response
        match = re.search(r"
```
python(.*)
```
", rsp, re.DOTALL)
        code_text = match.group(1).strip() if match else ""
        return code_text

class ExecuteCode(Action):
    async def run(self, code_path: str) -> str:
        # Dependent upon metagpt.tools.run_terminal_command
        prompt = f"""
        You are a code execution agent. Your task is to run the Python code located at {code_path}
        and return the output.
        Use the `run_terminal_command` tool.

        Command: python {code_path}

        Return the output of the command.
        """
        rsp = await self._aask(prompt)
        # Assuming the tool call will be made by the role based on the prompt
        return rsp

class ReadFile(Action):
    async def run(self, file_path: str) -> str:
        # Dependent upon metagpt.tools.read_file
        prompt = f"""
        You are a file reading agent. Your task is to read the content of the file located at {file_path}
        and return its content.
        Use the `read_file` tool.
        """
        rsp = await self._aask(prompt)
        return rsp

class CodingAgent(Role):
    name: str = "CodingAgent"
    profile: str = "Python Coder"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)        self.set_actions([WriteCode, ExecuteCode, ReadFile]) # Set the actions this agent can perform
        self.rc.todo = None

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        todo = self.rc.todo
        msg = self.rc.memory.get()[-1]

        if isinstance(todo, WriteCode):
            code_instruction = msg.content
            code_text = await WriteCode().run(instruction=code_instruction)

            # Save the code to a file
            file_path = "generated_code.py" # Define a default filename or get it from instruction
            write_file_instruction = f"Write the following Python code to a file named `{file_path}`:\n
```
python\n{code_text}\n
```
"
            # The Role needs to interact with tools or pass instructions for tool usage
            # For simplicity in this example, we'll assume the next action handles execution
            next_action = ExecuteCode(code_path=file_path) # Prepare the next action
            msg = Message(content=code_text, role=self.profile, cause_by=todo, send_to=next_action.name)
            self.rc.todo = next_action # Set the next action

        elif isinstance(todo, ExecuteCode):
            code_path_to_execute = todo.code_path # Get the code path from the action instance
            execution_result = await ExecuteCode(code_path=code_path_to_execute).run(code_path=code_path_to_execute)
            msg = Message(content=execution_result, role=self.profile, cause_by=todo)
            self.rc.todo = None # Task finished

        else:
            # Initial state or unknown task, assume it's a coding instruction
            code_instruction = msg.content
            self.rc.todo = WriteCode() # Start with writing code
            return await self._act() # Immediately process the WriteCode action

        return msg

if __name__ == "__main__":
    import asyncio

    async def main():
        # Example usage
        coding_agent = CodingAgent()
        task = "Write a python function that calculates the factorial of a number."
        await coding_agent.run(task)

        # After the first run (writing code), you would manually or programmatically
        # trigger the execution step if needed, based on the agent's response.
        # In a real application, the agent's internal logic or a manager role
        # would chain the actions.
        # For this simple example, let's assume the agent's _act handles the flow.

    asyncio.run(main())