from smolagents import CodeAgent, ToolCallingAgent, DuckDuckGoSearchTool, LiteLLMModel, PythonInterpreterTool, tool

from typing import Optional
import logging
import os
import time
import requests
from pathlib import Path
from urllib.parse import urlparse
import hashlib
import re
import sqlite3

logger = logging.getLogger(__name__)

SEARCH_QUERY= "agent"  # Replace with desired search term or topic
MAX_RESULTS= 50  # Adjust the number of papers you want to download
OUTPUT_FOLDER= "data"  # Folder to store downloaded papers
BASE_URL= "http://export.arxiv.org/api/query?"


@tool
def fetch_arxiv_papers(search_query:str = None, max_results: Optional[int]= None)->str:
    """
    Searches arXiv for research papers on a topic and saved teh papers to a folder
    Args:
        search_query: the topic to search for
        max_results: max results to return
    """

    # Create output folder if it doesn't exist
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # Fetch and parse papers
    print(f"Searching for papers on '{search_query}'...")
    response_text = _search_arxiv_papers(search_query, max_results)
    papers = _parse_paper_links(response_text)

    # Download each paper
    print(f"Found {len(papers)} papers. Starting download...")
    for title, pdf_link in papers:
        try:
            _download_paper(title, pdf_link, OUTPUT_FOLDER)
            time.sleep(2)  # Pause to avoid hitting rate limits
        except Exception as e:
            print(f"Failed to download '{title}': {e}")
    print("Download complete!")



def _get_papers_db():
    """Loads the papers database from a file."""
    db_name = "papers.db"
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            search_query TEXT,  -- Add search query column
            title TEXT UNIQUE,
            pdf_link TEXT,
            file_path TEXT,
            file_hash TEXT,
            file_content BLOB  -- Add file content column
        )
    """)
    conn.commit()
    return conn, cursor


def _sanitize_filename(title):
    """Sanitizes a string to be used as a filename."""
    # Remove any characters that are not alphanumeric, spaces, hyphens, or underscores
    return re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")


def _get_filename_from_url(url):
    # Parse the URL to get the path component
    parsed_url = urlparse(url)
    # Get the base name from the URL's path
    filename = os.path.basename(parsed_url.path)
    return filename

def _compute_file_hash(file_path, algorithm="sha256"):
    """Compute the hash of a file using the specified algorithm."""
    hash_func = hashlib.new(algorithm)

    with open(file_path, "rb") as file:
        # Read the file in chunks of 8192 bytes
        while chunk := file.read(8192):
            hash_func.update(chunk)

    return hash_func.hexdigest()

def _search_arxiv_papers(search_query, max_results=5):
    """Fetches metadata of papers from arXiv using the API."""
    url = f"{BASE_URL}search_query=all:{search_query}&start=0&max_results={max_results}"
    response = requests.get(url)
    response.raise_for_status()
    return response.text


def _parse_paper_links(response_text):
    """Parses paper links and titles from arXiv API response XML."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(response_text)
    papers = []
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        pdf_link = None
        for link in entry.findall("{http://www.w3.org/2005/Atom}link"):
            if link.attrib.get("title") == "pdf":
                pdf_link = link.attrib["href"] + ".pdf"
                break
        if pdf_link:
            title = _get_filename_from_url(pdf_link)
            print(title)
            papers.append((title, pdf_link))
    return papers


def _download_paper(title, pdf_link, output_folder):
    """Downloads a single paper PDF."""
    # Create a safe filename
    safe_title = _sanitize_filename(title)
    filename = os.path.join(output_folder, f"{safe_title}.pdf")
    response = requests.get(pdf_link, stream=True)
    response.raise_for_status()

    # Write the PDF to the specified folder
    with open(filename, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)
    print(f"Downloaded: {title}")




result = fetch_arxiv_papers(search_query="Cellular Automata")
print(result)


