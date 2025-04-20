# Deep Research AI Agent System Implementation
# Using LangGraph and LangChain with Tavily integration

import os
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime
from pydantic import BaseModel, Field

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import langgraph as lg
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

# Configuration and Models
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Define more capable model for complex reasoning
RESEARCH_MODEL = ChatOpenAI(model="gpt-4o", temperature=0.1)
# Use efficient model for intermediate tasks
ANALYSIS_MODEL = ChatOpenAI(model="gpt-4o", temperature=0.2)
# Use capable model for final drafting
DRAFTING_MODEL = ChatOpenAI(model="gpt-4o", temperature=0.7)

# State Management
class ResearchState(BaseModel):
    query: str = Field(description="The original research query")
    search_queries: List[str] = Field(default_factory=list, description="List of search queries to execute")
    research_plan: str = Field(default="", description="The plan for conducting the research")
    search_results: List[Dict] = Field(default_factory=list, description="Raw search results from Tavily")
    analyzed_content: Dict[str, Any] = Field(default_factory=dict, description="Analyzed and structured information")
    research_summary: str = Field(default="", description="Summary of research findings")
    draft_answer: str = Field(default="", description="Draft answer to the original query")
    final_answer: str = Field(default="", description="Final polished answer")
    metadata: Dict = Field(default_factory=dict, description="Metadata about the research process")
    
    class Config:
        arbitrary_types_allowed = True

# Tools

@tool
def tavily_search(query: str) -> List[Dict]:
    """Search the web using Tavily API and return results."""
    search = TavilySearchResults(k=5)
    results = search.invoke(query)
    return results

@tool
def extract_content_from_url(url: str) -> str:
    """Simulate extracting content from a URL (in a real implementation, 
    this would use a proper web scraper or Tavily's content extraction)"""
    # This would be replaced with actual content extraction logic
    return f"Extracted content from {url} would appear here."

# Agent Nodes

def research_planner(state: ResearchState) -> ResearchState:
    """Plan the research approach based on the query."""
    prompt = ChatPromptTemplate.from_template("""
    You are a research planning expert. Given the following query, develop a comprehensive 
    research plan and generate 3-5 specific search queries to gather information.
    
    Query: {query}
    
    Create a research plan that includes:
    1. Main aspects of the topic that need to be researched
    2. Potential credible sources to prioritize
    3. Specific angles to investigate
    
    Then, generate 3-5 specific search queries that will help gather comprehensive information.
    
    Return your response as a JSON with the following format:
    {{
        "research_plan": "detailed research plan here",
        "search_queries": ["query1", "query2", "query3", ...]
    }}
    """)
    
    response = RESEARCH_MODEL.invoke([HumanMessage(content=prompt.format(query=state.query))])
    
    try:
        result = json.loads(response.content)
        state.research_plan = result["research_plan"]
        state.search_queries = result["search_queries"]
    except (json.JSONDecodeError, KeyError):
        # Fallback handling in case of parsing issues
        state.research_plan = "General research on the topic"
        state.search_queries = [state.query]
    
    # Add metadata
    state.metadata["planning_timestamp"] = datetime.now().isoformat()
    
    return state

def execute_searches(state: ResearchState) -> ResearchState:
    """Execute search queries using Tavily."""
    all_results = []
    
    for query in state.search_queries:
        results = tavily_search(query)
        all_results.extend(results)
    
    # Remove duplicates by URL
    unique_results = []
    seen_urls = set()
    
    for result in all_results:
        if result.get("url") not in seen_urls:
            unique_results.append(result)
            seen_urls.add(result.get("url"))
    
    state.search_results = unique_results
    state.metadata["search_timestamp"] = datetime.now().isoformat()
    state.metadata["search_count"] = len(state.search_queries)
    state.metadata["results_count"] = len(unique_results)
    
    return state

def analyze_results(state: ResearchState) -> ResearchState:
    """Analyze and synthesize search results."""
    if not state.search_results:
        state.analyzed_content = {
            "key_findings": ["No search results were found."],
            "main_themes": [],
            "credibility_assessment": "No data to assess.",
            "information_gaps": ["Unable to find information on the query."]
        }
        state.research_summary = "No information was found on the specified query."
        return state
    
    # Prepare context for the analyzer
    search_results_context = "\n\n".join([
        f"Source {i+1}:\nTitle: {result.get('title', 'Untitled')}\n"
        f"URL: {result.get('url', 'No URL')}\n"
        f"Content: {result.get('content', 'No content')}"
        for i, result in enumerate(state.search_results[:10])  # Limit to first 10 results
    ])
    
    prompt = ChatPromptTemplate.from_template("""
    You are an expert research analyst. Analyze the following search results and extract 
    meaningful insights related to the original query.
    
    Original Query: {query}
    Research Plan: {research_plan}
    
    Search Results:
    {search_results}
    
    Provide your analysis in JSON format with the following structure:
    {{
        "key_findings": ["finding1", "finding2", ...],
        "main_themes": ["theme1", "theme2", ...],
        "credibility_assessment": "assessment of the credibility of sources",
        "information_gaps": ["gap1", "gap2", ...],
        "source_evaluation": [
            {{"source": "source name/url", "reliability": "high/medium/low", "relevance": "high/medium/low", "key_contribution": "what this source adds"}}
        ]
    }}
    
    Then provide a comprehensive research summary that synthesizes all findings.
    """)
    
    response = ANALYSIS_MODEL.invoke([
        HumanMessage(content=prompt.format(
            query=state.query,
            research_plan=state.research_plan,
            search_results=search_results_context
        ))
    ])
    
    # Extract JSON content (handles both raw JSON and JSON within markdown)
    try:
        json_content = response.content
        # If JSON is within markdown code blocks, extract it
        if "```json" in json_content:
            json_content = json_content.split("```json")[1].split("```")[0].strip()
        elif "```" in json_content:
            json_content = json_content.split("```")[1].split("```")[0].strip()
            
        analyzed_data = json.loads(json_content)
        state.analyzed_content = analyzed_data
        
        # Extract the research summary that should follow the JSON
        json_end_idx = response.content.find('}')
        if json_end_idx > -1 and json_end_idx + 1 < len(response.content):
            summary_text = response.content[json_end_idx + 1:].strip()
            # Remove any markdown or prefix text
            if "##" in summary_text or "Research Summary" in summary_text:
                for line in summary_text.split('\n'):
                    if line and not line.startswith('#') and not "Research Summary" in line:
                        state.research_summary += line + " "
            else:
                state.research_summary = summary_text
                
    except (json.JSONDecodeError, AttributeError) as e:
        # Fallback for parsing issues
        state.analyzed_content = {
            "key_findings": ["Error parsing analysis results."],
            "main_themes": ["Unable to determine main themes due to parsing error."],
            "credibility_assessment": "Analysis error",
            "information_gaps": ["Unable to identify information gaps due to analysis error."]
        }
        state.research_summary = "An error occurred during analysis: " + str(e)
    
    state.metadata["analysis_timestamp"] = datetime.now().isoformat()
    
    return state

def draft_answer(state: ResearchState) -> ResearchState:
    """Draft a comprehensive answer based on the research."""
    # Create context for the drafter
    findings = "\n".join([f"- {finding}" for finding in state.analyzed_content.get("key_findings", [])])
    themes = "\n".join([f"- {theme}" for theme in state.analyzed_content.get("main_themes", [])])
    gaps = "\n".join([f"- {gap}" for gap in state.analyzed_content.get("information_gaps", [])])
    
    prompt = ChatPromptTemplate.from_template("""
    You are an expert content creator specializing in comprehensive research summaries.
    Create a well-structured, informative answer to the original query based on the research findings.
    
    Original Query: {query}
    Research Summary: {research_summary}
    
    Key Findings:
    {findings}
    
    Main Themes:
    {themes}
    
    Information Gaps:
    {gaps}
    
    Draft a comprehensive, well-organized answer that addresses the original query.
    Include relevant details from the research while maintaining readability.
    Structure your response with appropriate headings, paragraphs and, when helpful, bullet points.
    Ensure the content flows logically and provides valuable insights.
    When appropriate, note any limitations in the available information.
    """)
    
    response = DRAFTING_MODEL.invoke([
        HumanMessage(content=prompt.format(
            query=state.query,
            research_summary=state.research_summary,
            findings=findings,
            themes=themes,
            gaps=gaps
        ))
    ])
    
    state.draft_answer = response.content
    state.metadata["drafting_timestamp"] = datetime.now().isoformat()
    
    return state

def finalize_answer(state: ResearchState) -> ResearchState:
    """Polish the draft answer and prepare the final response."""
    prompt = ChatPromptTemplate.from_template("""
    You are an expert editor with extensive experience in research communications.
    Review and refine the following draft answer to ensure it's comprehensive, accurate, 
    and effectively addresses the original query.
    
    Original Query: {query}
    
    Draft Answer:
    {draft_answer}
    
    Review this draft and create a final version that:
    1. Ensures all information is accurate and well-supported
    2. Improves clarity, flow, and readability
    3. Eliminates any redundancies or irrelevant information
    4. Adds any missing context or connections between ideas
    5. Maintains an authoritative but accessible tone
    6. Includes appropriate citations or references to sources where relevant
    7. Provides a concise conclusion that directly addresses the original query
    
    Return only the final polished answer.
    """)
    
    response = DRAFTING_MODEL.invoke([
        HumanMessage(content=prompt.format(
            query=state.query,
            draft_answer=state.draft_answer
        ))
    ])
    
    state.final_answer = response.content
    state.metadata["completion_timestamp"] = datetime.now().isoformat()
    
    return state

# Conditional Edge

def needs_more_research(state: ResearchState) -> str:
    """Determine if more research is needed based on analysis."""
    # Check if critical information gaps were identified
    gaps = state.analyzed_content.get("information_gaps", [])
    critical_gaps = [gap for gap in gaps if any(term in gap.lower() for term in 
                                               ["critical", "essential", "necessary", "important"])]
    
    # Check if we have enough key findings
    sufficient_findings = len(state.analyzed_content.get("key_findings", [])) >= 3
    
    # Check if the research summary expresses confidence
    confidence_indicators = ["insufficient", "limited", "not enough", "inadequate"]
    low_confidence = any(indicator in state.research_summary.lower() for indicator in confidence_indicators)
    
    if (critical_gaps or not sufficient_findings or low_confidence) and len(state.search_queries) < 10:
        # Generate additional queries based on gaps
        prompt = ChatPromptTemplate.from_template("""
        Based on the research conducted so far, we've identified some information gaps.
        Please generate 2-3 additional specific search queries to fill these gaps.
        
        Original Query: {query}
        Current Research Summary: {research_summary}
        
        Information Gaps:
        {gaps}
        
        Return only the additional search queries as a JSON array:
        ["query1", "query2", ...]
        """)
        
        gaps_text = "\n".join([f"- {gap}" for gap in gaps])
        response = RESEARCH_MODEL.invoke([
            HumanMessage(content=prompt.format(
                query=state.query,
                research_summary=state.research_summary,
                gaps=gaps_text
            ))
        ])
        
        try:
            # Extract JSON content
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            new_queries = json.loads(content)
            state.search_queries.extend(new_queries)
            return "additional_research"
        except (json.JSONDecodeError, IndexError):
            # If parsing fails, fall back to drafting with what we have
            return "draft"
    else:
        return "draft"

# Build the Graph

def build_research_graph() -> StateGraph:
    """Build the full research agent workflow graph."""
    # Initialize the graph
    graph = StateGraph(ResearchState)
    
    # Add nodes with unique identifiers
    graph.add_node("node_plan_research", research_planner)
    graph.add_node("node_execute_search", execute_searches)
    graph.add_node("node_analyze_results", analyze_results)
    graph.add_node("node_draft_research", draft_answer)
    graph.add_node("node_finalize_research", finalize_answer)
    
    # Connect the workflow
    graph.add_edge("node_plan_research", "node_execute_search")
    graph.add_edge("node_execute_search", "node_analyze_results")
    
    # Add conditional edge from analyze_results
    graph.add_conditional_edges(
        "node_analyze_results",
        needs_more_research,
        {
            "additional_research": "node_execute_search",
            "draft": "node_draft_research"
        }
    )
    
    graph.add_edge("node_draft_research", "node_finalize_research")
    graph.add_edge("node_finalize_research", END)
    
    # Set entry point
    graph.set_entry_point("node_plan_research")
    
    # Compile the graph
    return graph.compile()

# Main execution function
def run_research_agent(query: str) -> Dict[str, Any]:
    """Run the research agent pipeline on a given query."""
    # Create initial state
    state = ResearchState(query=query)
    
    # Create the graph
    research_graph = build_research_graph()
    
    # Create a memory saver for checkpointing
    memory_saver = MemorySaver()
    
    # Run the graph with the initial state
    result = research_graph.invoke(state)
    
    # Return the comprehensive result
    return {
        "final_answer": result["final_answer"],
        "metadata": {
            "execution_stats": result["metadata"],
            "research_plan": result["research_plan"],
            "search_queries_used": result["search_queries"],
            "search_results_count": len(result["search_results"]),
            "key_findings": result["analyzed_content"].get("key_findings", []),
            "information_gaps": result["analyzed_content"].get("information_gaps", [])
        }
    }

# Example usage
if __name__ == "__main__":
    query = "What are the latest developments in quantum computing and its potential impacts on cryptography?"
    result = run_research_agent(query)
    
    print("Research Results:")
    print("-" * 80)
    print(f"Query: {query}")
    print("-" * 80)
    print(result["final_answer"])
    print("-" * 80)
    print(f"Execution statistics: {json.dumps(result['metadata'], indent=2)}")