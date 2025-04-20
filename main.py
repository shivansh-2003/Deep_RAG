# Deep Research AI Agent System Implementation
# Using LangGraph and LangChain with Tavily and Perplexity integration

import os
import json
import asyncio
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pydantic import BaseModel, Field

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_perplexity import ChatPerplexity
import langgraph as lg
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from readability import Document
import dateutil.parser

# Load environment variables
load_dotenv()

# Configuration and Models
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Use Claude for primary reasoning and content analysis
CLAUDE_MODEL = ChatAnthropic(model="claude-3-7-sonnet-20250219", temperature=0.2)
# Fallback to GPT-4o for certain tasks
GPT_MODEL = ChatOpenAI(model="gpt-4o", temperature=0.2)
# Initialize Perplexity Chat
PERPLEXITY_MODEL = ChatPerplexity(api_key=PERPLEXITY_API_KEY, model="pplx-70b-online")

# Select primary model
MODEL = CLAUDE_MODEL

# State Management - single shared state across all agents
class ResearchState(BaseModel):
    query: str = Field(description="The original research query")
    research_plan: str = Field(default="", description="The plan for conducting the research")
    search_queries: List[str] = Field(default_factory=list, description="List of search queries to execute")
    requires_metrics: bool = Field(default=False, description="Whether the query explicitly requires quantitative metrics")
    perplexity_results: Dict[str, Any] = Field(default_factory=dict, description="Results from Perplexity API")
    search_results: List[Dict] = Field(default_factory=list, description="Raw search results from Tavily")
    content_details: List[Dict] = Field(default_factory=list, description="Full content extracted from URLs")
    analyzed_content: Dict[str, Any] = Field(default_factory=dict, description="Synthesized information and findings")
    draft_answer: str = Field(default="", description="Draft answer to the original query")
    verified_info: Dict[str, Any] = Field(default_factory=dict, description="Fact-checked information")
    final_answer: str = Field(default="", description="Final polished answer")
    citations: List[Dict[str, str]] = Field(default_factory=list, description="Tracked citations for sources")
    metadata: Dict = Field(default_factory=dict, description="Process metadata and timestamps")

# Tools
@tool
def tavily_search(query: str, time_range: str = "auto", search_depth: str = "advanced") -> List[Dict]:
    """Search the web using Tavily API with configurable parameters.
    
    Args:
        query: The search query
        time_range: Time range for results (auto, day, week, month, year)
        search_depth: Depth of search (basic or advanced)
    
    Returns:
        List of search results
    """
    search = TavilySearchResults(
        api_key=TAVILY_API_KEY,
        k=7,
        include_domains=[],
        exclude_domains=[],
        search_depth=search_depth,
        include_raw_content=True,
        include_images=False,
        max_results=7,
    )
    results = search.invoke({"query": query, "search_depth": search_depth})
    return results

@tool
def tavily_extract_content(url: str) -> Dict[str, Any]:
    """Extract content from a URL using Tavily's content extraction API.
    
    Args:
        url: The URL to extract content from
        
    Returns:
        Dictionary with extracted content and metadata
    """
    headers = {"Content-Type": "application/json", "x-api-key": TAVILY_API_KEY}
    payload = {"url": url, "include_images": False}
    
    try:
        response = requests.post(
            "https://api.tavily.com/extract",
            headers=headers,
            json=payload,
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            return {
                "url": url,
                "title": data.get("title", ""),
                "text": data.get("content", ""),
                "publish_date": data.get("publish_date", ""),
                "status": "success"
            }
        else:
            return {
                "url": url, 
                "status": "failed", 
                "error": f"Status code: {response.status_code}"
            }
    except Exception as e:
        return {"url": url, "status": "failed", "error": str(e)}

@tool
def perplexity_search(query: str, focus: str = "normal") -> Dict[str, Any]:
    """Search the web using Perplexity API for comprehensive summaries.
    
    Args:
        query: The search query
        focus: Focus mode (normal, concise, comprehensive)
        
    Returns:
        Dictionary with Perplexity response including text and sources
    """
    if not PERPLEXITY_API_KEY:
        return {
            "text": "Perplexity API key not configured. Skipping this step.",
            "sources": [],
            "status": "failed"
        }
    
    try:
        # Use the LangChain ChatPerplexity integration
        response = PERPLEXITY_MODEL.invoke(query)
        
        # Extract source information if available
        sources = []
        if hasattr(response, 'additional_kwargs') and 'sources' in response.additional_kwargs:
            sources = response.additional_kwargs['sources']
        
        return {
            "text": response.content,
            "sources": sources,
            "status": "success"
        }
    except Exception as e:
        return {
            "text": f"Error: {str(e)}",
            "sources": [],
            "status": "failed"
        }

@tool
def extract_content_from_url(url: str) -> str:
    """Extract content from a URL using BeautifulSoup with readability enhancement.
    Used as fallback when Tavily extraction isn't available.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            # First use readability to get the main content
            doc = Document(response.text)
            title = doc.title()
            readable_content = doc.summary()
            
            # Then extract text from the readability HTML
            soup = BeautifulSoup(readable_content, 'html.parser')
            
            # Get text content
            text = soup.get_text(separator='\n', strip=True)
            
            # Attempt to extract publish date
            try:
                publish_date = ""
                date_meta = soup.select_one('meta[property="article:published_time"]')
                if date_meta:
                    publish_date = date_meta.get('content', '')
                else:
                    # Try other common date meta tags
                    date_candidates = [
                        soup.select_one('meta[name="date"]'),
                        soup.select_one('meta[name="pubdate"]'),
                        soup.select_one('time'),
                    ]
                    for candidate in date_candidates:
                        if candidate:
                            date_str = candidate.get('content') or candidate.get('datetime') or ''
                            if date_str:
                                publish_date = date_str
                                break
            except:
                publish_date = ""
            
            # Truncate if too long but keep more content for Claude's larger context window
            max_length = 15000
            if len(text) > max_length:
                text = text[:max_length] + "... [content truncated]"
                
            return {
                "title": title,
                "text": text,
                "publish_date": publish_date,
                "status": "success"
            }
        return {
            "title": "",
            "text": f"Failed to retrieve content: Status code {response.status_code}",
            "publish_date": "",
            "status": "failed"
        }
    except Exception as e:
        return {
            "title": "",
            "text": f"Error extracting content: {str(e)}",
            "publish_date": "",
            "status": "failed"
        }

# Agent Nodes

def planner_agent(state: ResearchState) -> ResearchState:
    """Plan the research strategy and generate search queries with special focus on metrics if needed."""
    # First, analyze if the query requires quantitative metrics
    metrics_check_prompt = ChatPromptTemplate.from_template("""
    Analyze the following query and determine if it explicitly asks for quantitative metrics,
    numbers, statistics, or comparative data.
    
    Query: {query}
    
    Return your analysis as a JSON:
    {{
        "requires_metrics": true/false,
        "metric_subjects": ["subject1", "subject2"],  // What entities need metrics
        "metric_types": ["type1", "type2"]  // What kinds of metrics are needed (parameters, benchmarks, etc.)
    }}
    """)
    
    response = MODEL.invoke([HumanMessage(content=metrics_check_prompt.format(query=state.query))])
    
    try:
        # Extract JSON content
        json_content = response.content
        if "```json" in json_content:
            json_content = json_content.split("```json")[1].split("```")[0].strip()
        elif "```" in json_content:
            json_content = json_content.split("```")[1].split("```")[0].strip()
            
        metrics_analysis = json.loads(json_content)
        state.requires_metrics = metrics_analysis.get("requires_metrics", False)
        state.metadata["metric_subjects"] = metrics_analysis.get("metric_subjects", [])
        state.metadata["metric_types"] = metrics_analysis.get("metric_types", [])
    except:
        # Fallback
        state.requires_metrics = False
    
    # Now generate the research plan and queries
    planning_prompt = ChatPromptTemplate.from_template("""
    You are a research planning expert. Given the following query, develop a comprehensive
    research plan and generate specific search queries to gather information.
    
    Query: {query}
    
    {metrics_instruction}
    
    First, create a detailed research plan that includes:
    1. Main aspects of the topic that need to be researched
    2. Key information that must be gathered
    3. Specific angles to investigate
    
    Then, generate these specific search queries:
    1. One broad query to get an overview of the topic
    2. 3-5 targeted queries for specific aspects mentioned in the query
    {metrics_queries}
    
    Return your response as a JSON with the following format:
    {{
        "research_plan": "detailed research plan here",
        "search_queries": ["query1", "query2", "query3", ...]
    }}
    """)
    
    # Add special instructions if metrics are required
    metrics_instruction = ""
    metrics_queries = ""
    if state.requires_metrics:
        metrics_instruction = """
        This query explicitly asks for quantitative metrics, numbers, and/or comparative data.
        Your research plan MUST focus on finding specific numerical data, benchmarks, and statistics.
        """
        
        metrics_queries = """
        3. 2-3 queries specifically aimed at finding numerical data, using terms like:
           - "[subject] performance metrics"
           - "[subject] benchmarks"
           - "[subject] parameters comparison"
           - "[subject] technical specifications"
        4. Add site-specific searches for data sources like:
           - "site:paperswithcode.com [subject] benchmarks"
           - "site:huggingface.co/docs [subject] parameters"
        """
    
    response = MODEL.invoke([
        HumanMessage(content=planning_prompt.format(
            query=state.query,
            metrics_instruction=metrics_instruction,
            metrics_queries=metrics_queries
        ))
    ])
    
    try:
        # Extract JSON content
        json_content = response.content
        if "```json" in json_content:
            json_content = json_content.split("```json")[1].split("```")[0].strip()
        elif "```" in json_content:
            json_content = json_content.split("```")[1].split("```")[0].strip()
            
        result = json.loads(json_content)
        state.research_plan = result["research_plan"]
        state.search_queries = result["search_queries"]
    except:
        # Fallback handling
        state.research_plan = "General research on the topic"
        state.search_queries = [state.query]
    
    state.metadata["planning_timestamp"] = datetime.now().isoformat()
    state.metadata["requires_metrics"] = state.requires_metrics
    
    return state

def search_agent(state: ResearchState) -> ResearchState:
    """Execute search queries using Tavily and Perplexity with intelligent routing."""
    all_results = []
    
    # First, get a broad overview from Perplexity for the main query
    if PERPLEXITY_API_KEY:
        state.metadata["search_strategy"] = "hybrid_perplexity_tavily"
        
        # Use Perplexity for the main query to get an overview
        main_query = state.query
        perplexity_result = perplexity_search(main_query)
        
        state.perplexity_results = perplexity_result
        
        # Extract cited URLs from Perplexity to avoid duplicating searches
        perplexity_urls = set()
        for source in perplexity_result.get("sources", []):
            if isinstance(source, dict) and "url" in source:
                perplexity_urls.add(source["url"])
            elif isinstance(source, str) and (source.startswith("http://") or source.startswith("https://")):
                perplexity_urls.add(source)
                
        # Add source information to citations tracking
        for i, source in enumerate(perplexity_result.get("sources", [])):
            if isinstance(source, dict):
                title = source.get("title", "Untitled")
                url = source.get("url", "")
            elif isinstance(source, str) and (source.startswith("http://") or source.startswith("https://")):
                title = f"Source {i+1}"
                url = source
            else:
                continue
            
            state.citations.append({
                "id": f"p{i+1}",
                "title": title,
                "url": url,
                "source_type": "perplexity"
            })
    else:
        state.metadata["search_strategy"] = "tavily_only"
    
    # Use Tavily for specific queries
    for i, query in enumerate(state.search_queries):
        # Execute the search
        results = tavily_search({"query": query, "time_range": "auto", "search_depth": "advanced"})
        
        # Add to results, marking which query produced them
        for result in results:
            result["query_source"] = query
            result["query_index"] = i
            all_results.append(result)
    
    # Remove duplicates by URL
    unique_results = []
    seen_urls = set()
    
    for result in all_results:
        url = result.get("url")
        if url and url not in seen_urls:
            # Skip URLs already found in Perplexity results to avoid duplication
            if url in perplexity_urls:
                continue
                
            unique_results.append(result)
            seen_urls.add(url)
    
    state.search_results = unique_results
    state.metadata["search_timestamp"] = datetime.now().isoformat()
    state.metadata["tavily_results_count"] = len(unique_results)
    state.metadata["perplexity_sources_count"] = len(state.perplexity_results.get("sources", []))
    
    return state

def content_extraction_agent(state: ResearchState) -> ResearchState:
    """Extract full content from search result URLs with parallel processing."""
    content_details = []
    
    # Add Perplexity summary as a special source
    if state.perplexity_results and state.perplexity_results.get("status") == "success":
        content_details.append({
            "url": "perplexity_summary",
            "title": "Perplexity Summary",
            "snippet": state.perplexity_results.get("text", "")[:500] + "...",
            "full_content": state.perplexity_results.get("text", ""),
            "publish_date": datetime.now().isoformat(),
            "source_type": "perplexity_summary"
        })
    
    # Process sources from Perplexity
    perplexity_sources = state.perplexity_results.get("sources", [])
    perplexity_urls = []
    
    for source in perplexity_sources:
        if "url" in source and "title" in source:
            perplexity_urls.append({
                "url": source["url"],
                "title": source.get("title", "Untitled"),
                "snippet": source.get("snippet", ""),
                "source_type": "perplexity"
            })
    
    # Combine Perplexity and Tavily sources, prioritizing sources that might have metrics
    # for metric queries
    all_sources = []
    
    # First add Perplexity sources
    all_sources.extend(perplexity_urls)
    
    # Then add top Tavily results
    for result in state.search_results[:7]:  # Limit to top 7 results
        all_sources.append({
            "url": result.get("url"),
            "title": result.get("title", "Untitled"),
            "snippet": result.get("content", ""),
            "source_type": "tavily"
        })
    
    # For metrics queries, prioritize sources that appear to contain data
    if state.requires_metrics:
        # Prioritize sources that mention metrics, benchmarks, etc.
        data_terms = ["benchmark", "metric", "performance", "comparison", "parameter", 
                     "statistic", "score", "accuracy", "evaluation", "leaderboard"]
        
        for source in all_sources:
            source["priority"] = 0
            snippet = source.get("snippet", "").lower()
            title = source.get("title", "").lower()
            
            # Check for data indicators in title and snippet
            for term in data_terms:
                if term in title:
                    source["priority"] += 3
                if term in snippet:
                    source["priority"] += 1
                    
            # Extra boost for HuggingFace, PapersWithCode and other data-rich sources
            url = source.get("url", "").lower()
            if any(domain in url for domain in ["huggingface", "paperswithcode", "leaderboard", 
                                              "benchmark", "kaggle", "github"]):
                source["priority"] += 5
        
        # Sort by priority for metrics queries
        all_sources.sort(key=lambda x: x.get("priority", 0), reverse=True)
    
    # Function to extract content from a single URL
    def extract_single_url(source_info):
        url = source_info["url"]
        title = source_info["title"]
        snippet = source_info["snippet"]
        source_type = source_info["source_type"]
        
        # Skip the perplexity summary as it's already processed
        if url == "perplexity_summary":
            return None
            
        try:
            # Try Tavily extraction first
            extraction_result = tavily_extract_content(url)
            
            # If Tavily extraction failed, fall back to regular extraction
            if extraction_result.get("status") == "failed":
                extraction_result = extract_content_from_url(url)
                
            # Ensure we have the title, even if extraction failed
            if not extraction_result.get("title"):
                extraction_result["title"] = title
                
            # Add source metadata
            extraction_result["url"] = url
            extraction_result["snippet"] = snippet
            extraction_result["source_type"] = source_type
            
            return extraction_result
        except Exception as e:
            # Return error information
            return {
                "url": url,
                "title": title,
                "snippet": snippet,
                "full_content": f"Error extracting content: {str(e)}",
                "status": "failed",
                "source_type": source_type
            }
    
    # Process URLs in parallel to improve performance
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Limit to top 12 sources (combined from Perplexity and Tavily)
        sources_to_process = all_sources[:12]
        future_to_url = {executor.submit(extract_single_url, source): source for source in sources_to_process}
        
        for future in concurrent.futures.as_completed(future_to_url):
            source = future_to_url[future]
            try:
                result = future.result()
                if result:  # Skip None results (like the perplexity summary)
                    # Add full content extraction results
                    content_details.append({
                        "url": result.get("url"),
                        "title": result.get("title", source.get("title", "Untitled")),
                        "snippet": source.get("snippet", ""),
                        "full_content": result.get("text", ""),
                        "publish_date": result.get("publish_date", ""),
                        "source_type": source.get("source_type"),
                        "status": result.get("status", "unknown")
                    })
                    
                    # Add to citations if not already there
                    url = result.get("url")
                    if url and not any(citation["url"] == url for citation in state.citations):
                        citation_id = f"t{len(state.citations) + 1}"
                        state.citations.append({
                            "id": citation_id,
                            "title": result.get("title", "Untitled"),
                            "url": url,
                            "source_type": source.get("source_type")
                        })
            except Exception as e:
                print(f"Error processing {source.get('url')}: {e}")
    
    state.content_details = content_details
    state.metadata["extraction_timestamp"] = datetime.now().isoformat()
    state.metadata["extracted_sources_count"] = len(content_details)
    
    return state

def analysis_agent(state: ResearchState) -> ResearchState:
    """Synthesize and evaluate the extracted information with focus on metrics for queries requiring them."""
    if not state.content_details:
        state.analyzed_content = {
            "key_findings": ["No content was found."],
            "information_gaps": ["Unable to find information on the query."]
        }
        return state
    
    # Prepare context for analysis, prioritizing successful extractions
    successful_sources = [s for s in state.content_details if s.get("status") != "failed"]
    failed_sources = [s for s in state.content_details if s.get("status") == "failed"]
    
    # Use all successful sources and add a note about failed ones
    sources_to_analyze = successful_sources
    if failed_sources:
        failed_note = f"Note: {len(failed_sources)} sources could not be accessed."
    else:
        failed_note = ""
    
    # Prepare the context with source content
    # For metrics queries, we want to preserve more numerical data in the context
    max_content_per_source = 2500 if state.requires_metrics else 1000
    
    content_context = "\n\n".join([
        f"Source {i+1} [ID:{state.citations[i]['id'] if i < len(state.citations) else 'unknown'}]:\n"
        f"Title: {item['title']}\n"
        f"URL: {item['url']}\n"
        f"Date: {item.get('publish_date', 'Unknown')}\n"
        f"Content Summary: {item['snippet']}\n"
        f"Full Content Excerpt: {item['full_content'][:max_content_per_source]}..."
        for i, item in enumerate(sources_to_analyze)
    ])
    
    # Add failed sources note
    if failed_note:
        content_context += f"\n\n{failed_note}"
    
    # Create the appropriate prompt based on query type
    if state.requires_metrics:
        prompt = ChatPromptTemplate.from_template("""
        You are an expert research analyst specializing in quantitative data analysis. 
        The user query explicitly asks for metrics, numbers, and quantitative comparisons.
        Analyze the following content focusing on extracting specific metrics, numbers, benchmarks,
        parameters, and quantitative data to answer the query.
        
        Original Query: {query}
        Research Plan: {research_plan}
        
        Content to Analyze:
        {content}
        
        Provide your analysis in JSON format with the following structure:
        {{
            "key_findings": [
                "finding1 (with specific metrics where available)",
                "finding2 (with specific metrics where available)",
                ...
            ],
            "quantitative_data": [
                {{
                    "entity": "Name of entity (e.g., model name)",
                    "metric_name": "Name of metric (e.g., MMLU score)",
                    "value": "Numerical or formatted value", 
                    "unit": "Unit if applicable",
                    "source_id": "ID of source that provided this data"
                }},
                ...
            ],
            "comparative_analysis": "Analysis comparing the metrics across entities",
            "main_themes": ["theme1", "theme2", ...],
            "information_gaps": ["gap1", "gap2", ...],
            "source_assessment": "Assessment of source quality and data reliability"
        }}
        
        IMPORTANT:
        1. Extract ALL specific numbers, metrics, and quantitative data found in the sources
        2. When listing metrics, always cite which source provided each data point
        3. If sources give conflicting metrics, note the discrepancy and provide both with sources
        4. Include specific values wherever possible (e.g., "GPT-4 has 1.7 trillion parameters" is better than "GPT-4 has many parameters")
        """)
    else:
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
        
        When extracting findings:
        1. Include specific data points and numbers when available
        2. Note any conflicting information between sources
        3. Prioritize recent information when dates are available
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
    
    # Check for missing data that would be important for the query
    missing_aspects_prompt = ChatPromptTemplate.from_template("""
    Based on the original query and the key findings so far, identify any important aspects 
    of the query that have not been addressed or require additional information.
    
    Original Query: {query}
    
    Key Findings So Far:
    {findings}
    
    Information Gaps Already Identified:
    {gaps}
    
    Identify any missing aspects that are important to fully answer the query.
    Return as a JSON array of missing aspects:
    ["missing aspect 1", "missing aspect 2", ...]
    
    If no important aspects are missing, return an empty array.
    """)
    
    findings = "\n".join([f"- {finding}" for finding in state.analyzed_content.get("key_findings", [])])
    gaps = "\n".join([f"- {gap}" for gap in state.analyzed_content.get("information_gaps", [])])
    
    response = MODEL.invoke([
        HumanMessage(content=missing_aspects_prompt.format(
            query=state.query,
            findings=findings,
            gaps=gaps
        ))
    ])
    
    try:
        # Extract missing aspects
        missing_content = response.content
        if "```json" in missing_content:
            missing_content = missing_content.split("```json")[1].split("```")[0].strip()
        elif "```" in missing_content:
            missing_content = missing_content.split("```")[1].split("```")[0].strip()
        
        missing_aspects = json.loads(missing_content)
        state.analyzed_content["missing_aspects"] = missing_aspects
    except:
        state.analyzed_content["missing_aspects"] = []
    
    state.metadata["analysis_timestamp"] = datetime.now().isoformat()
    return state

def drafting_agent(state: ResearchState) -> ResearchState:
    """Compose a draft answer based on the analyzed information with proper citations and metrics."""
    key_findings = "\n".join([f"- {finding}" for finding in state.analyzed_content.get("key_findings", [])])
    information_gaps = "\n".join([f"- {gap}" for gap in state.analyzed_content.get("information_gaps", [])])
    
    # Format citations for reference
    citations_info = "\n".join([
        f"[{cite['id']}] {cite['title']} ({cite['url']})"
        for cite in state.citations
    ])
    
    # Create the appropriate prompt based on query type
    if state.requires_metrics:
        # Get quantitative data if available
        quant_data = state.analyzed_content.get("quantitative_data", [])
        formatted_quant_data = json.dumps(quant_data, indent=2) if quant_data else "No specific quantitative data extracted."
        
        comparative_analysis = state.analyzed_content.get("comparative_analysis", "")
        
        prompt = ChatPromptTemplate.from_template("""
        You are an expert content creator specializing in data-driven reports. Draft a comprehensive 
        answer to the original query with strong emphasis on quantitative data and metrics.
        
        Original Query: {query}
        
        Key Findings:
        {key_findings}
        
        Quantitative Data:
        {quant_data}
        
        Comparative Analysis:
        {comparative_analysis}
        
        Information Gaps:
        {information_gaps}
        
        Available Citations:
        {citations}
        
        Draft a well-structured, data-rich answer that:
        1. Addresses the query directly, emphasizing numerical data and metrics
        2. Organizes information logically, possibly using tables or structured formats for metrics
        3. Includes proper citations using the citation IDs provided (format: [ID])
        4. Clearly presents comparative analysis between entities
        5. Notes significant limitations or information gaps
        6. Uses a clear, authoritative tone appropriate for technical/data content
        
        The final answer MUST include specific numbers, metrics, and quantitative data with citations.
        When including a metric, ALWAYS add the citation ID in format [ID].
        """)
        
        response = MODEL.invoke([
            HumanMessage(content=prompt.format(
                query=state.query,
                key_findings=key_findings,
                quant_data=formatted_quant_data,
                comparative_analysis=comparative_analysis,
                information_gaps=information_gaps,
                citations=citations_info
            ))
        ])
    else:
        # Standard prompt for non-metric queries
        prompt = ChatPromptTemplate.from_template("""
        You are an expert content creator. Draft a comprehensive answer to the original query 
        based on the research findings.
        
        Original Query: {query}
        
        Key Findings:
        {key_findings}
        
        Information Gaps:
        {information_gaps}
        
        Available Citations:
        {citations}
        
        Draft a well-structured answer that:
        1. Addresses the query directly
        2. Organizes information in a logical flow
        3. Includes proper citations using the citation IDs provided (format: [ID])
        4. Notes any significant limitations in the available information
        5. Uses a clear, informative style
        
        When including important facts or claims, add the citation ID in format [ID].
        """)
        
        response = MODEL.invoke([
            HumanMessage(content=prompt.format(
                query=state.query,
                key_findings=key_findings,
                information_gaps=information_gaps,
                citations=citations_info
            ))
        ])
    
    state.draft_answer = response.content
    state.metadata["drafting_timestamp"] = datetime.now().isoformat()
    return state

def fact_checking_agent(state: ResearchState) -> ResearchState:
    """Verify claims and sources in the draft answer with special attention to quantitative claims."""
    # Prepare context with key findings and quantitative data
    key_findings = "\n".join([f"- {finding}" for finding in state.analyzed_content.get("key_findings", [])])
    
    # Add quantitative data context if available
    quant_data_context = ""
    if state.requires_metrics and "quantitative_data" in state.analyzed_content:
        quant_data = state.analyzed_content["quantitative_data"]
        formatted_data = [
            f"- {item.get('entity', 'Unknown entity')}: {item.get('metric_name', 'Metric')}: "
            f"{item.get('value', 'Unknown value')} {item.get('unit', '')}"
            f" (Source: {item.get('source_id', 'unknown')})"
            for item in quant_data
        ]
        quant_data_context = "\n".join(formatted_data)
    
    # Prepare citations context
    citations_context = "\n".join([
        f"[{cite['id']}] {cite['title']} ({cite['url']})"
        for cite in state.citations
    ])
    
    # Create appropriate prompt based on query type
    if state.requires_metrics:
        prompt = ChatPromptTemplate.from_template("""
        You are a fact-checking expert specializing in data verification. Review the draft answer 
        against the research findings with special attention to quantitative claims.
        
        Original Query: {query}
        
        Draft Answer:
        {draft_answer}
        
        Key Findings from Research:
        {key_findings}
        
        Quantitative Data from Research:
        {quant_data}
        
        Citations Used:
        {citations}
        
        Verify the following aspects of the draft:
        1. ALL numerical claims are accurate and match the source data
        2. ALL metrics, statistics, and quantities are correctly cited
        3. Comparisons between entities are mathematically correct
        4. No important quantitative findings were omitted
        5. Citations are properly used
        
        Return your assessment in JSON format:
        {{
            "accuracy_assessment": "overall assessment of factual accuracy",
            "number_verification": [
                {{
                    "claim": "exact numerical claim from draft",
                    "verification": "verified/incorrect/unsupported",
                    "correction": "correction if needed",
                    "note": "verification note"
                }},
                ...
            ],
            "missing_metrics": ["important metric 1 that was omitted", ...],
            "citation_errors": ["description of citation error 1", ...],
            "suggested_corrections": ["correction 1", ...]
        }}
        """)
        
        response = MODEL.invoke([
            HumanMessage(content=prompt.format(
                query=state.query,
                draft_answer=state.draft_answer,
                key_findings=key_findings,
                quant_data=quant_data_context,
                citations=citations_context
            ))
        ])
    else:
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
        
        Citations Used:
        {citations}
        
        Return your assessment in JSON format:
        {{
            "accuracy_assessment": "overall assessment of factual accuracy",
            "unsupported_claims": ["claim1", "claim2", ...],
            "suggested_corrections": ["correction1", "correction2", ...],
            "citation_errors": ["citation error 1", ...],
            "verification_notes": "additional verification notes"
        }}
        """)
        
        response = MODEL.invoke([
            HumanMessage(content=prompt.format(
                query=state.query,
                draft_answer=state.draft_answer,
                key_findings=key_findings,
                citations=citations_context
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
    """Polish and finalize the answer based on fact checking and format appropriately."""
    # Prepare different contexts based on query type
    if state.requires_metrics:
        # For metrics queries, include the verification of numerical claims
        number_verification = state.verified_info.get("number_verification", [])
        formatted_verification = "\n".join([
            f"- Claim: \"{item.get('claim', '')}\"\n  Status: {item.get('verification', '')}\n  "
            f"Correction: {item.get('correction', 'None')}" 
            for item in number_verification
        ]) if number_verification else "No specific numerical claims verified."
        
        missing_metrics = "\n".join([f"- {metric}" for metric in state.verified_info.get("missing_metrics", [])])
        
        prompt = ChatPromptTemplate.from_template("""
        You are an expert editor specializing in data-driven content. Refine the draft answer 
        based on fact-checking feedback with special attention to numerical accuracy.
        
        Original Query: {query}
        
        Draft Answer:
        {draft_answer}
        
        Fact-Checking Assessment:
        Accuracy: {accuracy}
        
        Numerical Verification:
        {number_verification}
        
        Missing Metrics:
        {missing_metrics}
        
        Citation Errors:
        {citation_errors}
        
        Suggested Corrections:
        {corrections}
        
        Create a final polished answer that:
        1. Addresses all fact-checking concerns, especially numerical accuracy
        2. Presents quantitative data clearly, possibly using tables or structured lists
        3. Includes proper in-text citations for all claims and metrics
        4. Maintains clarity and readability
        5. Directly answers the original query with emphasis on metrics and comparisons
        6. Notes any important limitations in the available information
        
        Add a citation reference section at the end with all cited sources listed.
        Format: [ID] Title - URL
        
        IMPORTANT: Ensure ALL numbers and metrics are correct and properly cited.
        Use markdown formatting for readability (headers, lists, emphasis, tables).
        """)
        
        citation_errors = "\n".join([f"- {error}" for error in state.verified_info.get("citation_errors", [])])
        corrections = "\n".join([f"- {correction}" for correction in state.verified_info.get("suggested_corrections", [])])
        
        response = MODEL.invoke([
            HumanMessage(content=prompt.format(
                query=state.query,
                draft_answer=state.draft_answer,
                accuracy=state.verified_info.get("accuracy_assessment", "Unknown"),
                number_verification=formatted_verification,
                missing_metrics=missing_metrics,
                citation_errors=citation_errors,
                corrections=corrections
            ))
        ])
    else:
        # Standard editing prompt for non-metric queries
        unsupported_claims = "\n".join([f"- {claim}" for claim in state.verified_info.get("unsupported_claims", [])])
        corrections = "\n".join([f"- {correction}" for correction in state.verified_info.get("suggested_corrections", [])])
        citation_errors = "\n".join([f"- {error}" for error in state.verified_info.get("citation_errors", [])])
        
        prompt = ChatPromptTemplate.from_template("""
        You are an expert editor. Refine the draft answer based on fact-checking feedback.
        
        Original Query: {query}
        
        Draft Answer:
        {draft_answer}
        
        Fact-Checking Assessment:
        Accuracy: {accuracy}
        Unsupported Claims: {unsupported_claims}
        Suggested Corrections: {corrections}
        Citation Errors: {citation_errors}
        
        Create a final polished answer that:
        1. Addresses all fact-checking concerns
        2. Maintains clarity and readability
        3. Directly answers the original query
        4. Notes any important limitations in the available information
        5. Includes proper citations
        
        Add a citation reference section at the end with all cited sources listed.
        Format: [ID] Title - URL
        
        Use markdown formatting for readability (headers, lists, emphasis).
        """)
        
        response = MODEL.invoke([
            HumanMessage(content=prompt.format(
                query=state.query,
                draft_answer=state.draft_answer,
                accuracy=state.verified_info.get("accuracy_assessment", "Unknown"),
                unsupported_claims=unsupported_claims,
                corrections=corrections,
                citation_errors=citation_errors
            ))
        ])
    
    # Add reflection for comprehensiveness
    answer = response.content
    
    reflection_prompt = ChatPromptTemplate.from_template("""
    Review this answer for completeness. Does it fully address the user's query?
    
    Original Query: {query}
    
    Final Answer:
    {answer}
    
    Is there anything significant missing that would make this answer more complete?
    If yes, add ONLY what's missing. If no, return the answer unchanged.
    
    Return only the final complete answer.
    """)
    
    reflection_response = MODEL.invoke([
        HumanMessage(content=reflection_prompt.format(
            query=state.query,
            answer=answer
        ))
    ])
    
    state.final_answer = reflection_response.content
    state.metadata["completion_timestamp"] = datetime.now().isoformat()
    
    return state

# Conditional Edge Function
def needs_more_research(state: ResearchState) -> str:
    """Determine if additional research is needed based on information gaps and missing aspects."""
    # Check if critical information gaps were identified
    gaps = state.analyzed_content.get("information_gaps", [])
    missing_aspects = state.analyzed_content.get("missing_aspects", [])
    
    # Combine regular gaps with missing aspects
    all_gaps = gaps + missing_aspects
    
    # Check for critical gaps (those that seem important)
    critical_gap_indicators = ["critical", "essential", "necessary", "important", "key", 
                               "significant", "crucial", "primary", "main", "major"]
    
    critical_gaps = [gap for gap in all_gaps if any(term in gap.lower() for term in critical_gap_indicators)]
    
    # For metrics queries, check if we have enough quantitative data
    if state.requires_metrics:
        quant_data = state.analyzed_content.get("quantitative_data", [])
        has_sufficient_metrics = len(quant_data) >= 3  # Arbitrary threshold
        
        metric_gaps = [gap for gap in all_gaps if any(term in gap.lower() for term in 
                                                     ["metric", "number", "statistic", "figure",
                                                      "quantitative", "data", "benchmark"])]
        
        if (not has_sufficient_metrics or metric_gaps) and len(state.search_queries) < 10:
            # Generate metric-focused queries
            prompt = ChatPromptTemplate.from_template("""
            Based on the current research, we need more specific metrics and quantitative data.
            Generate 2-3 additional search queries focused on finding numerical data about:
            
            Original Query: {query}
            
            Missing Metrics/Data: {gaps}
            
            For each missing metric or data point, create a search query specifically designed 
            to find that numerical information. Format the queries to target reliable sources
            of benchmarks and metrics.
            
            Return only the additional search queries as a JSON array:
            ["query1", "query2", ...]
            """)
            
            gaps_text = "\n".join([f"- {gap}" for gap in all_gaps])
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
    
    # Standard check for non-metric queries  
    if critical_gaps and len(state.search_queries) < 8:
        # Generate additional search queries based on gaps
        prompt = ChatPromptTemplate.from_template("""
        Based on the current research, generate 2-3 additional search queries to fill these information gaps:
        
        Original Query: {query}
        
        Information Gaps: {gaps}
        
        Return only the additional search queries as a JSON array:
        ["query1", "query2", ...]
        """)
        
        gaps_text = "\n".join([f"- {gap}" for gap in all_gaps])
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
    """Build the research agent workflow graph with improved routing."""
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
    print(f"Starting research on: {query}")
    start_time = datetime.now()
    
    # Initialize state with query
    state = ResearchState(query=query)
    
    # Build and run the graph
    research_graph = build_research_graph()
    memory_saver = MemorySaver()
    result = research_graph.invoke(state)
    
    end_time = datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    
    # Format citations for return
    citations = []
    for citation in result["citations"]:
        citations.append({
            "id": citation["id"],
            "title": citation["title"],
            "url": citation["url"]
        })
    
    # Return final result with enhanced metadata
    return {
        "query": query,
        "final_answer": result["final_answer"],
        "metadata": {
            "execution_time": {
                "started": result["metadata"].get("planning_timestamp", ""),
                "completed": result["metadata"].get("completion_timestamp", ""),
                "seconds_elapsed": execution_time
            },
            "search_strategy": result["metadata"].get("search_strategy", ""),
            "search_queries": result["search_queries"],
            "requires_metrics": result["metadata"].get("requires_metrics", False),
            "sources_count": {
                "perplexity": result["metadata"].get("perplexity_sources_count", 0),
                "tavily": result["metadata"].get("tavily_results_count", 0),
                "total_extracted": result["metadata"].get("extracted_sources_count", 0)
            },
            "key_findings_count": len(result["analyzed_content"].get("key_findings", [])),
            "citations": citations
        }
    }

# Example usage
if __name__ == "__main__":
    query = "ok can you tell me the information about claude 3.7 sonnet and its new advanced features"
    result = run_research_agent(query)
    
    print("\n" + "=" * 50)
    print(f"QUERY: {query}")
    print("=" * 50)
    print(result["final_answer"])
    print("\n" + "=" * 50)
    print(f"Execution took {result['metadata']['execution_time']['seconds_elapsed']:.2f} seconds")
    print(f"Sources used: {result['metadata']['sources_count']['total_extracted']} from Perplexity ({result['metadata']['sources_count']['perplexity']}) and Tavily ({result['metadata']['sources_count']['tavily']})")
    if result['metadata']['requires_metrics']:
        print("Query was identified as requiring quantitative metrics.")
    print("=" * 50)





    