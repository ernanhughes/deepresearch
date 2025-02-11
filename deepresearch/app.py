import argparse
import json
import os
import threading
import mimetypes
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import datasets
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login
import gradio as gr

from scripts.reformulator import prepare_response
from scripts.run_agents import get_single_file_description, get_zip_description
from scripts.text_inspector_tool import TextInspectorTool
from scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SearchInformationTool,
    SimpleTextBrowser,
    VisitTool,
)
from scripts.visual_qa import visualizer
from smolagents import (
    CodeAgent,
    LiteLLMModel,
    ToolCallingAgent,
)
from smolagents.agent_types import AgentText, AgentImage, AgentAudio
from smolagents.gradio_ui import pull_messages_from_step, handle_agent_output_types

# Load environment variables
load_dotenv(override=True)
login(os.getenv("HF_TOKEN"))

# Set constants
AUTHORIZED_IMPORTS = [
    "requests", "zipfile", "os", "pandas", "numpy", "sympy", "json",
    "bs4", "pubchempy", "xml", "yahoo_finance", "Bio", "sklearn",
    "scipy", "pydub", "io", "PIL", "chess", "PyPDF2", "pptx", "torch",
    "datetime", "fractions", "csv"
]

SET = "validation"
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 " \
             "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"

BROWSER_CONFIG = {
    "viewport_size": 1024 * 5,
    "downloads_folder": "downloads_folder",
    "request_kwargs": {"headers": {"User-Agent": user_agent}, "timeout": 300},
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),
}

os.makedirs(BROWSER_CONFIG['downloads_folder'], exist_ok=True)

# Load dataset
eval_ds = datasets.load_dataset("gaia-benchmark/GAIA", "2023_all")[SET]
eval_ds = eval_ds.rename_columns({"Question": "question", "Final answer": "true_answer", "Level": "task"})

def preprocess_file_paths(row):
    if len(row["file_name"]) > 0:
        row["file_name"] = f"data/gaia/{SET}/" + row["file_name"]
    return row

eval_ds = eval_ds.map(preprocess_file_paths)
eval_df = pd.DataFrame(eval_ds)
print("Loaded evaluation dataset:")
print(eval_df["task"].value_counts())

# Initialize model
model = LiteLLMModel(model_id="ollama/qwen2.5-coder:14b", api_base="http://localhost:11434")

ti_tool = TextInspectorTool(model, 20000)
browser = SimpleTextBrowser(**BROWSER_CONFIG)

WEB_TOOLS = [
    SearchInformationTool(browser), VisitTool(browser), PageUpTool(browser),
    PageDownTool(browser), FinderTool(browser), FindNextTool(browser),
    ArchiveSearchTool(browser), TextInspectorTool(model, 20000)
]

agent = CodeAgent(
    model=model,
    tools=[visualizer] + WEB_TOOLS,
    max_steps=5,
    verbosity_level=2,
    additional_authorized_imports=AUTHORIZED_IMPORTS,
)

document_inspection_tool = TextInspectorTool(model, 20000)

def stream_to_gradio(agent, task: str):
    """Runs an agent with the given task and streams responses."""
    for step_log in agent.run(task, stream=True):
        for message in pull_messages_from_step(step_log):
            yield message
    final_answer = handle_agent_output_types(step_log)
    yield gr.ChatMessage(role="assistant", content=str(final_answer))

class GradioUI:
    """A one-line interface to launch the agent in Gradio."""
    def __init__(self, agent, file_upload_folder: Optional[str] = None):
        self.agent = agent
        self.file_upload_folder = file_upload_folder
        if file_upload_folder and not os.path.exists(file_upload_folder):
            os.mkdir(file_upload_folder)

    def interact_with_agent(self, prompt, messages):
        messages.append(gr.ChatMessage(role="user", content=prompt))
        yield messages
        for msg in stream_to_gradio(self.agent, task=prompt):
            messages.append(msg)
            yield messages

    def launch(self, **kwargs):
        with gr.Blocks() as demo:
            gr.Markdown("""# AI Agent Interface

Try out our AI-powered research assistant!
""")
            stored_messages = gr.State([])
            chatbot = gr.Chatbot(label="AI Assistant", type="messages", resizeable=True)
            text_input = gr.Textbox(lines=1, label="Your request")
            text_input.submit(self.interact_with_agent, [text_input, stored_messages], [chatbot])
        demo.launch(debug=True, share=True, **kwargs)

GradioUI(agent).launch()
