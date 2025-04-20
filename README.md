# Deep Research AI Agent 🧠🔍

## Comprehensive Web Research System with Advanced LLM Integration

A sophisticated, state-of-the-art research assistant that combines multiple language models and information retrieval systems to provide thorough, fact-checked, and well-cited answers to complex queries.

![Version](https://img.shields.io/badge/Version-1.0-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🌟 Core Capabilities

The Deep Research AI Agent is designed to:

- Execute comprehensive web research using multiple data sources
- Generate data-rich, thoroughly cited responses
- Provide special handling for queries requiring quantitative metrics
- Implement rigorous fact-checking and verification
- Deliver polished, well-structured answers with proper citation formatting

---

## 🏗️ Technical Architecture

### Multi-Agent System Design

This system leverages LangGraph for orchestrating a complex workflow of specialized agents, each handling a distinct phase of the research process:

#### 1. **Planner Agent**
- Analyzes query requirements and develops comprehensive research strategies
- Identifies queries requiring quantitative metrics with specialized handling
- Generates targeted search queries for different aspects of the research topic
- Creates detailed research plans outlining key investigation areas

#### 2. **Search Agent**
- Implements hybrid information retrieval combining multiple data sources
- Executes searches through Tavily API for detailed web results
- Leverages Perplexity API for broader contextual summaries
- Avoids duplicate sources and optimizes search strategy based on query type
- Manages API-specific parameters (search depth, time range)

#### 3. **Content Extraction Agent**
- Executes parallel processing of multiple URLs for efficient data retrieval
- Primary extraction through Tavily content extraction API
- Fallback extraction using BeautifulSoup with readability enhancements
- Handles content truncation for large documents
- Preserves metadata like publication dates

#### 4. **Analysis Agent**
- Synthesizes extracted content into cohesive findings
- Special handling for queries requiring metrics:
  - Extracts specific numerical data points
  - Creates structured quantitative data collections
  - Performs comparative analysis between metrics
- Identifies information gaps and missing aspects
- Assesses source quality and reliability

#### 5. **Drafting Agent**
- Creates well-structured, comprehensive answer drafts
- Incorporates proper citations using consistent ID format
- Emphasizes numerical data for metric-focused queries
- Organizes information in logical flow

#### 6. **Fact-Checking Agent**
- Verifies factual accuracy of the draft answer
- Special rigor for numerical claims and quantitative data
- Identifies unsupported claims and citation errors
- Provides detailed verification assessment

#### 7. **Finalizing Agent**
- Refines and polishes the answer based on fact-checking feedback
- Ensures comprehensive coverage of the original query
- Applies markdown formatting for readability
- Creates properly formatted citation reference section

### Dynamic Research Flow Logic

The system incorporates intelligent routing:

- **Conditional Research Loop**: Dynamically determines if additional research is needed
- **Metric-Focused Iteration**: Generates specialized follow-up queries for missing quantitative data
- **Gap Analysis**: Identifies and addresses critical information gaps

### Foundation Models & APIs

- **Primary LLM**: Claude 3.7 Sonnet (for main reasoning and content generation)
- **Secondary LLM**: GPT-4o (fallback for certain tasks)
- **Search APIs**:
  - Tavily Search API (precision web search)
  - Perplexity API (comprehensive summaries)

---

## 💻 Technical Implementation

### Core Technologies

- **LangGraph**: Graph-based workflow orchestration
- **LangChain**: Foundation for agent implementation and API integration
- **Pydantic**: Type-safe state management
- **BeautifulSoup & Readability**: Content extraction
- **Asyncio & Concurrent.futures**: Parallel processing

### State Management

The entire research process uses a shared `ResearchState` object containing:

```python
class ResearchState(BaseModel):
    query: str
    research_plan: str
    search_queries: List[str]
    requires_metrics: bool
    perplexity_results: Dict[str, Any]
    search_results: List[Dict]
    content_details: List[Dict]
    analyzed_content: Dict[str, Any]
    draft_answer: str
    verified_info: Dict[str, Any]
    final_answer: str
    citations: List[Dict[str, str]]
    metadata: Dict
```

### Tool Implementations

The system implements various tools:

- `tavily_search`: Configurable web search through Tavily API
- `tavily_extract_content`: Content extraction from URLs
- `perplexity_search`: Comprehensive summaries via Perplexity
- `extract_content_from_url`: Fallback extraction with readability enhancement

### Specialized Handling for Quantitative Queries

The system detects queries requiring numerical data and:

1. Creates metric-focused search queries
2. Prioritizes sources likely to contain quantitative data
3. Implements specialized extraction patterns for numerical information
4. Applies rigorous verification to numerical claims
5. Formats metrics clearly in the final output

---

## 🚀 Getting Started

### Prerequisites

The following API keys are required:

- `TAVILY_API_KEY`
- `PERPLEXITY_API_KEY`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/deep-research-ai-agent.git
cd deep-research-ai-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file with your API keys
echo "TAVILY_API_KEY=your_key_here" > .env
echo "PERPLEXITY_API_KEY=your_key_here" >> .env
echo "OPENAI_API_KEY=your_key_here" >> .env
echo "ANTHROPIC_API_KEY=your_key_here" >> .env
```

### Basic Usage

```python
from main import run_research_agent

# Run a research query
result = run_research_agent("What are the latest advancements in quantum computing?")

# Print the final answer
print(result["final_answer"])

# Access metadata about the research process
print(f"Time taken: {result['metadata']['execution_time']['seconds_elapsed']} seconds")
print(f"Sources used: {result['metadata']['sources_count']['total_extracted']}")
```

### Advanced Usage Example

```python
# For queries explicitly requiring quantitative metrics
result = run_research_agent(
    "Compare the performance of the latest LLM models including parameters counts, benchmark scores, and training data size"
)

# The system will automatically detect the need for metrics and adapt its research approach
```

---

## 📊 System Output

The system returns a structured dictionary containing:

1. **final_answer**: The comprehensive, well-formatted response
2. **metadata**: Detailed information about the research process:
   - Execution time statistics
   - Search strategy used
   - Search queries executed
   - Source counts (Perplexity, Tavily)
   - Whether metrics were required
   - Key findings count
   - Full citation list

---

## 🔧 Customization Options

### Model Selection

The system allows choosing between different foundation models:

```python
# Use Claude as primary model (default)
MODEL = CLAUDE_MODEL

# Or use GPT-4o instead
MODEL = GPT_MODEL
```

### Search Parameters

Tavily search parameters can be adjusted:

```python
# In tavily_search tool
search = TavilySearchResults(
    api_key=TAVILY_API_KEY,
    k=7,  # Number of results
    search_depth="advanced",  # "basic" or "advanced"
    include_domains=[],  # Restrict to specific domains
    exclude_domains=[],  # Exclude specific domains
)
```

---

## ⚠️ Limitations & Considerations

- **API Dependency**: Performance relies on external API availability and rate limits
- **Research Scope**: Limited to information available via the integrated search APIs
- **Temporal Limitations**: Bound by the knowledge cutoff dates of the underlying LLMs
- **Content Extraction Challenges**: Some websites may have anti-scraping measures
- **Processing Time**: Comprehensive research may take several minutes to complete

---

## 🔮 Future Enhancements

- Integration with academic paper repositories for research queries
- Support for image and chart analysis
- Domain-specific research agents (medical, legal, technical)
- Interactive research mode with user feedback incorporation
- Custom data source integration capabilities
- Support for long-term memory across research sessions

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgements

- [Anthropic](https://www.anthropic.com/) for Claude API
- [OpenAI](https://openai.com/) for GPT API
- [Tavily](https://tavily.com/) for search API
- [Perplexity](https://www.perplexity.ai/) for research API
- [LangChain](https://langchain.com/) and [LangGraph](https://python.langchain.com/docs/langgraph) development teams

---

*Built with ❤️ for advanced research automation*