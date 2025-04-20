# Deep Research AI Agent System Implementation
# Using LangGraph and LangChain with Tavily integration

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import langgraph as lg
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()

# Configuration and Models
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
MODEL = ChatOpenAI(model="gpt-4o", temperature=0.2)

# State Management - single shared state across all agents
class ResearchState(BaseModel):
    query: str = Field(description="The original research query")
    research_plan: str = Field(default="", description="The plan for conducting the research")
    search_queries: List[str] = Field(default_factory=list, description="List of search queries to execute")
    search_results: List[Dict] = Field(default_factory=list, description="Raw search results from Tavily")
    content_details: List[Dict] = Field(default_factory=list, description="Full content extracted from URLs")
    analyzed_content: Dict[str, Any] = Field(default_factory=dict, description="Synthesized information and findings")
    draft_answer: str = Field(default="", description="Draft answer to the original query")
    verified_info: Dict[str, Any] = Field(default_factory=dict, description="Fact-checked information")
    final_answer: str = Field(default="", description="Final polished answer")
    metadata: Dict = Field(default_factory=dict, description="Process metadata and timestamps")

# Tools
@tool
def tavily_search(query: str) -> List[Dict]:
    """Search the web using Tavily API and return results."""
    search = TavilySearchResults(k=5)
    results = search.invoke(query)
    return results

@tool
def extract_content_from_url(url: str) -> str:
    """Extract content from a URL using BeautifulSoup."""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            # Get text
            text = soup.get_text(separator='\n', strip=True)
            # Break into lines and remove leading/trailing spaces
            lines = (line.strip() for line in text.splitlines())
            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # Drop blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)
            # Truncate if too long
            if len(text) > 8000:
                text = text[:8000] + "... [content truncated]"
            return text
        return f"Failed to retrieve content: Status code {response.status_code}"
    except Exception as e:
        return f"Error extracting content: {str(e)}"

# Agent Nodes

def planner_agent(state: ResearchState) -> ResearchState:
    """Plan the research strategy and generate search queries."""
    prompt = ChatPromptTemplate.from_template("""
    You are a research planning expert. Given the following query, develop a research plan 
    and generate 3-5 specific search queries to gather comprehensive information.
    
    Query: {query}
    
    Return your response as a JSON with the following format:
    {{
        "research_plan": "detailed research plan here",
        "search_queries": ["query1", "query2", "query3"]
    }}
    """)
    
    response = MODEL.invoke([HumanMessage(content=prompt.format(query=state.query))])
    
    try:
        result = json.loads(response.content)
        state.research_plan = result["research_plan"]
        state.search_queries = result["search_queries"]
    except:
        # Fallback handling
        state.research_plan = "General research on the topic"
        state.search_queries = [state.query]
    
    state.metadata["planning_timestamp"] = datetime.now().isoformat()
    return state

def search_agent(state: ResearchState) -> ResearchState:
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
    return state

def content_extraction_agent(state: ResearchState) -> ResearchState:
    """Extract full content from search result URLs."""
    content_details = []
    
    # Process only top results to avoid excessive processing
    for result in state.search_results[:5]:
        url = result.get("url")
        if url:
            extracted_content = extract_content_from_url(url)
            content_details.append({
                "url": url,
                "title": result.get("title", "Untitled"),
                "snippet": result.get("content", ""),
                "full_content": extracted_content
            })
    
    state.content_details = content_details
    state.metadata["extraction_timestamp"] = datetime.now().isoformat()
    return state

def analysis_agent(state: ResearchState) -> ResearchState:
    """Synthesize and evaluate the extracted information."""
    if not state.content_details:
        state.analyzed_content = {
            "key_findings": ["No content was found."],
            "information_gaps": ["Unable to find information on the query."]
        }
        return state
    
    # Prepare context for analysis
    content_context = "\n\n".join([
        f"Source {i+1}:\nTitle: {item['title']}\nURL: {item['url']}\n"
        f"Content Summary: {item['snippet']}\n"
        f"Full Content Excerpt: {item['full_content'][:500]}..."
        for i, item in enumerate(state.content_details)
    ])
    
    prompt = ChatPromptTemplate.from_template("""
    You are an expert research analyst. Analyze the following content and extract 
    key insights related to the original query.
    
    Original Query: {query}
    Research Plan: {research_plan}
    
    Content to Analyze:
    {content}
    
    Provide your analysis in JSON format with the following structure:
    {{
        "key_findings": ["finding1", "finding2", ...],
        "main_themes": ["theme1", "theme2", ...],
        "information_gaps": ["gap1", "gap2", ...],
        "source_assessment": "brief assessment of source quality"
    }}
    """)
    
    response = MODEL.invoke([
        HumanMessage(content=prompt.format(
            query=state.query,
            research_plan=state.research_plan,
            content=content_context
        ))
    ])
    
    try:
        # Extract JSON content
        json_content = response.content
        if "```json" in json_content:
            json_content = json_content.split("```json")[1].split("```")[0].strip()
        elif "```" in json_content:
            json_content = json_content.split("```")[1].split("```")[0].strip()
            
        state.analyzed_content = json.loads(json_content)
    except:
        # Fallback if parsing fails
        state.analyzed_content = {
            "key_findings": ["Analysis failed to parse results properly."],
            "main_themes": [],
            "information_gaps": ["Technical error in analysis phase."],
            "source_assessment": "Unable to assess sources due to processing error."
        }
    
    state.metadata["analysis_timestamp"] = datetime.now().isoformat()
    return state

def drafting_agent(state: ResearchState) -> ResearchState:
    """Compose a draft answer based on the analyzed information."""
    key_findings = "\n".join([f"- {finding}" for finding in state.analyzed_content.get("key_findings", [])])
    information_gaps = "\n".join([f"- {gap}" for gap in state.analyzed_content.get("information_gaps", [])])
    
    prompt = ChatPromptTemplate.from_template("""
    You are an expert content creator. Draft a comprehensive answer to the original query 
    based on the research findings.
    
    Original Query: {query}
    
    Key Findings:
    {key_findings}
    
    Information Gaps:
    {information_gaps}
    
    Draft a well-structured answer that addresses the query directly. 
    Use a clear, informative style. Note any significant limitations in the available information.
    """)
    
    response = MODEL.invoke([
        HumanMessage(content=prompt.format(
            query=state.query,
            key_findings=key_findings,
            information_gaps=information_gaps
        ))
    ])
    
    state.draft_answer = response.content
    state.metadata["drafting_timestamp"] = datetime.now().isoformat()
    return state

def fact_checking_agent(state: ResearchState) -> ResearchState:
    """Verify claims and sources in the draft answer."""
    prompt = ChatPromptTemplate.from_template("""
    You are a fact-checking expert. Review the draft answer against the key findings and source content.
    Identify any claims that:
    1. Are not supported by the sources
    2. Contradict the sources
    3. Need qualification or additional context
    
    Original Query: {query}
    
    Draft Answer:
    {draft_answer}
    
    Key Findings from Research:
    {key_findings}
    
    Return your assessment in JSON format:
    {{
        "accuracy_assessment": "overall assessment of factual accuracy",
        "unsupported_claims": ["claim1", "claim2", ...],
        "suggested_corrections": ["correction1", "correction2", ...],
        "verification_notes": "additional verification notes"
    }}
    """)
    
    key_findings = "\n".join([f"- {finding}" for finding in state.analyzed_content.get("key_findings", [])])
    
    response = MODEL.invoke([
        HumanMessage(content=prompt.format(
            query=state.query,
            draft_answer=state.draft_answer,
            key_findings=key_findings
        ))
    ])
    
    try:
        # Extract JSON content
        json_content = response.content
        if "```json" in json_content:
            json_content = json_content.split("```json")[1].split("```")[0].strip()
        elif "```" in json_content:
            json_content = json_content.split("```")[1].split("```")[0].strip()
            
        state.verified_info = json.loads(json_content)
    except:
        state.verified_info = {
            "accuracy_assessment": "Unable to complete fact checking due to parsing error.",
            "unsupported_claims": [],
            "suggested_corrections": [],
            "verification_notes": "Technical error occurred during verification."
        }
    
    state.metadata["fact_checking_timestamp"] = datetime.now().isoformat()
    return state

def finalizing_agent(state: ResearchState) -> ResearchState:
    """Polish and finalize the answer based on fact checking."""
    prompt = ChatPromptTemplate.from_template("""
    You are an expert editor. Refine the draft answer based on fact-checking feedback.
    
    Original Query: {query}
    
    Draft Answer:
    {draft_answer}
    
    Fact-Checking Assessment:
    Accuracy: {accuracy}
    Unsupported Claims: {unsupported_claims}
    Suggested Corrections: {corrections}
    
    Create a final polished answer that:
    1. Addresses all fact-checking concerns
    2. Maintains clarity and readability
    3. Directly answers the original query
    4. Notes any important limitations in the available information
    5. Includes citations where appropriate
    
    Return only the final polished answer.
    """)
    
    unsupported_claims = "\n".join([f"- {claim}" for claim in state.verified_info.get("unsupported_claims", [])])
    corrections = "\n".join([f"- {correction}" for correction in state.verified_info.get("suggested_corrections", [])])
    
    response = MODEL.invoke([
        HumanMessage(content=prompt.format(
            query=state.query,
            draft_answer=state.draft_answer,
            accuracy=state.verified_info.get("accuracy_assessment", "Unknown"),
            unsupported_claims=unsupported_claims,
            corrections=corrections
        ))
    ])
    
    state.final_answer = response.content
    state.metadata["completion_timestamp"] = datetime.now().isoformat()
    return state

# Conditional Edge Function
def needs_more_research(state: ResearchState) -> str:
    """Determine if additional research is needed based on information gaps."""
    # Check if critical information gaps were identified
    gaps = state.analyzed_content.get("information_gaps", [])
    critical_gaps = [gap for gap in gaps if any(term in gap.lower() for term in 
                                              ["critical", "essential", "necessary", "important"])]
    
    # Check if we have enough key findings
    sufficient_findings = len(state.analyzed_content.get("key_findings", [])) >= 3
    
    if critical_gaps and not sufficient_findings and len(state.search_queries) < 8:
        # Generate additional search queries based on gaps
        prompt = ChatPromptTemplate.from_template("""
        Based on the current research, generate 2-3 additional search queries to fill these information gaps:
        
        Original Query: {query}
        Information Gaps: {gaps}
        
        Return only the additional search queries as a JSON array:
        ["query1", "query2", ...]
        """)
        
        gaps_text = "\n".join([f"- {gap}" for gap in gaps])
        response = MODEL.invoke([
            HumanMessage(content=prompt.format(
                query=state.query,
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
            return "needs_more_research"
        except:
            return "proceed_to_draft"
    else:
        return "proceed_to_draft"

# Graph Construction
def build_research_graph() -> StateGraph:
    """Build the research agent workflow graph."""
    graph = StateGraph(ResearchState)
    
    # Add nodes
    graph.add_node("planner", planner_agent)
    graph.add_node("search", search_agent)
    graph.add_node("content_extraction", content_extraction_agent)
    graph.add_node("analysis", analysis_agent)
    graph.add_node("drafting", drafting_agent)
    graph.add_node("fact_checking", fact_checking_agent)
    graph.add_node("finalizing", finalizing_agent)
    
    # Connect nodes with directed edges
    graph.add_edge("planner", "search")
    graph.add_edge("search", "content_extraction")
    graph.add_edge("content_extraction", "analysis")
    
    # Add conditional edge from analysis
    graph.add_conditional_edges(
        "analysis",
        needs_more_research,
        {
            "needs_more_research": "search",
            "proceed_to_draft": "drafting"
        }
    )
    
    graph.add_edge("drafting", "fact_checking")
    graph.add_edge("fact_checking", "finalizing")
    graph.add_edge("finalizing", END)
    
    # Set entry point
    graph.set_entry_point("planner")
    
    return graph.compile()

# Main execution function
def run_research_agent(query: str) -> Dict[str, Any]:
    """Run the research agent pipeline on a given query."""
    # Initialize state with query
    state = ResearchState(query=query)
    
    # Build and run the graph
    research_graph = build_research_graph()
    memory_saver = MemorySaver()
    result = research_graph.invoke(state)
    
    # Return final result
    return {
        "query": query,
        "final_answer": result["final_answer"],
        "metadata": {
            "execution_time": {
                "started": result["metadata"].get("planning_timestamp", ""),
                "completed": result["metadata"].get("completion_timestamp", "")
            },
            "search_queries": result["search_queries"],
            "key_findings": result["analyzed_content"].get("key_findings", []),
            "information_gaps": result["analyzed_content"].get("information_gaps", [])
        }
    }

# Example usage
if __name__ == "__main__":
    query = "ok can you tell me the current comptetitors of the avrious llmsof the companies an provide brief info about their most quantifiable metrics"
    result = run_research_agent(query)
    
    print("\n" + "=" * 50)
    print(f"QUERY: {query}")
    print("=" * 50)
    print(result["final_answer"])
    print("=" * 50)