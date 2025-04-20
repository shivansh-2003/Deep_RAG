# Deep Research Agent

A modular AI agent system that conducts comprehensive web research on any topic by coordinating multiple specialized agents.

## Architecture

The system follows a multi-agent architecture:

1. **Planner Agent**: Creates a research strategy and search queries
2. **Search Agent**: Retrieves information from the web (using Tavily)
3. **Content Extraction Agent**: Fetches full content from search result URLs
4. **Analysis Agent**: Synthesizes and evaluates information
5. **Drafting Agent**: Composes a draft answer
6. **Fact-Checking Agent**: Verifies claims and sources
7. **Finalizing Agent**: Polishes and finalizes the answer

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

## Usage

```python
from main import run_research_agent

query = "What are the latest advancements in renewable energy storage?"
result = run_research_agent(query)

print(result["final_answer"])
```

You can also run the example directly:

```
python main.py
```

## Customization

- Modify agent prompts in the main.py file
- Adjust the graph structure to add or remove agents
- Change the model parameters as needed 