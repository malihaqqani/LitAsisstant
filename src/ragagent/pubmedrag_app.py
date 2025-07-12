"""
Chainlit App - Frontend"""

try:
    import chainlit as cl
except ImportError:
    cl = None  # type: ignore

import os
import sys
import asyncio
import logging
from typing import Dict, Any, List
from dotenv import load_dotenv
from openai import OpenAI



try:
    from ragagent.core.pubmedrag_core import QuestionDrivenRAG
except ImportError:
    # Fallback for relative import
    from .core.pubmedrag_core import QuestionDrivenRAG

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global RAG instance per session
rag_instances: Dict[str, QuestionDrivenRAG] = {}


def format_citations(citations: List[Dict[str, Any]]) -> str:
    """Format citations for display."""
    if not citations:
        return ""
    
    formatted = "\n\n**References:**\n"
    for i, citation in enumerate(citations, 1):
        pmid = citation.get('pmid', '')
        title = citation.get('title', 'No title')
        authors = citation.get('authors', 'Unknown authors')
        journal = citation.get('journal', '')
        pub_date = citation.get('pub_date', '')
        doi = citation.get('doi', '')
        
        # Format the citation
        formatted += f"\n{i}. **{title}**\n"
        if authors:
            formatted += f"   *Authors:* {authors}\n"
        if journal:
            formatted += f"   *Journal:* {journal}"
            if pub_date:
                formatted += f" ({pub_date})"
            formatted += "\n"
        if pmid:
            formatted += f"   *PMID:* [{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)\n"
        if doi:
            formatted += f"   *DOI:* [{doi}](https://doi.org/{doi})\n"
    
    return formatted


@cl.on_chat_start
async def start():
    """Initialize the PubMedRAG session."""
    
    # Get user info
    user = cl.user_session.get("user")
    user_id = user.id if user else "anonymous"
    
    # Check for required API keys
    openai_key = os.getenv("LLM_API_KEY")
    email = os.getenv("PUBMED_EMAIL")
    
    # Debug logging for API key issues
    logger.info(f"🔍 Debug - OpenAI Key present: {bool(openai_key)}")
    if openai_key:
        logger.info(f"🔍 Debug - OpenAI Key length: {len(openai_key)}")
        logger.info(f"🔍 Debug - OpenAI Key starts with: {openai_key[:10]}...")
        logger.info(f"🔍 Debug - OpenAI Key ends with: ...{openai_key[-10:]}")
    logger.info(f"🔍 Debug - Email present: {bool(email)}")
    
    if not openai_key:
        await cl.Message(
            content="❌ **OpenAI API Key Missing**\n\nPlease set your OPENAI_API_KEY environment variable:\n1. Create a .env file in the project root\n2. Add: OPENAI_API_KEY=your-api-key-here\n3. Get your key from: https://platform.openai.com/"
        ).send()
        return
    
    if not email:
        await cl.Message(
            content="❌ **Email Missing**\n\nPlease set your PUBMED_EMAIL environment variable:\n1. Add to your .env file: PUBMED_EMAIL=your-email@example.com\n2. This is required by NCBI for PubMed API access"
        ).send()
        return
    
    # DeepSeek Configuration (much cheaper than OpenAI!)
    llm_base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    llm_model = os.getenv("LLM_MODEL", "deepseek-chat")
    ncbi_api_key = os.getenv("NCBI_API_KEY")  # Optional but recommended
    
    # More debug logging
    logger.info(f"🔍 Debug - LLM Base URL: {llm_base_url}")
    logger.info(f"🔍 Debug - LLM Model: {llm_model}")
    logger.info(f"🔍 Debug - NCBI Key present: {bool(ncbi_api_key)}")
    
    try:
        # Initialize PubMedRAG
        rag = QuestionDrivenRAG(
            email=email,
            llm_api_key=openai_key,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            ncbi_api_key=ncbi_api_key,
            temperature=0.3
        )
        
        # Store RAG instance
        session_id = rag.session_id
        rag_instances[session_id] = rag
        cl.user_session.set("rag_session_id", session_id)
        cl.user_session.set("user_id", user_id)
        
        # Create welcome message
        welcome_content = f"""🔬I'm a biomedical research assistant! I can help you with: 
Ask me anything about biomedical research, genetics, diseases, treatments, or scientific discoveries!
"""
        
        # Create action buttons with proper payload field
        actions = [
            cl.Action(name="session_info", value="info", description="📊 Session Info", payload={"action": "session_info"}),
            cl.Action(name="new_session", value="new", description="🆕 New Session", payload={"action": "new_session"}),
        ]
        
        await cl.Message(content=welcome_content, actions=actions).send()
        
    except Exception as e:
        logger.error(f"Failed to initialize PubMedRAG: {e}")
        await cl.Message(
            content=f"❌ **Initialization Error**\n\nFailed to initialize PubMedRAG: {str(e)}\n\nPlease check your configuration and try again."
        ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle user messages with PubMedRAG."""
    
    # Get RAG instance
    session_id = cl.user_session.get("rag_session_id")
    if not session_id or session_id not in rag_instances:
        await cl.Message(
            content="❌ **Session Error**\n\nRAG session not found. Please refresh the page to start a new session."
        ).send()
        return
    
    rag = rag_instances[session_id]
    user_message = message.content
    
    # Create a streaming message
    msg = cl.Message(content="")
    await msg.send()
    
    try:
        # Show initial status
        await msg.stream_token("🔍 **Analyzing your question...**\n\n")
        
        # Process the question with RAG
        result = await asyncio.to_thread(
            rag.answer_question, 
            user_message, 
            k=15  # Retrieve more chunks for better context
        )
        
        answer = result.get("answer", "No answer generated.")
        citations = result.get("citations", [])
        search_performed = result.get("search_performed", False)
        new_articles = result.get("new_articles", 0)
        total_articles = result.get("total_articles", 0)
        
        # Clear the initial status and stream the answer
        msg.content = ""
        
        # Add search status if new search was performed
        if search_performed:
            await msg.stream_token(f"📊 **New Literature Search Performed**\n")
            await msg.stream_token(f"Found {new_articles} new articles. Total indexed: {total_articles}\n\n")
        
        await msg.stream_token("🤖 **Answer:**\n\n")
        await msg.stream_token(answer)
        
        # Add formatted citations
        if citations:
            citations_text = format_citations(citations)
            await msg.stream_token(citations_text)
        else:
            await msg.stream_token("\n\n*No specific citations were referenced in this answer.*")
        
        # Add session info
        session_info = rag.get_session_info()
        await msg.stream_token(f"\n\n---\n*Session: {session_info['total_questions']} questions, {session_info['total_articles']} articles indexed*")
        
        await msg.update()
        
    except asyncio.TimeoutError:
        await msg.stream_token("⏰ **Request timed out.** Please try a simpler question or check your internet connection.")
        await msg.update()
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        
        if "api_key" in str(e).lower() or "openai" in str(e).lower():
            error_msg = f"❌ **OpenAI API Error:** {str(e)}\n\nPlease check your API key and try again."
        elif "email" in str(e).lower() or "entrez" in str(e).lower():
            error_msg = f"❌ **PubMed API Error:** {str(e)}\n\nPlease check your email configuration for PubMed access."
        else:
            error_msg = f"❌ **Error:** {str(e)}\n\nPlease try rephrasing your question or contact support if the issue persists."
        
        await msg.stream_token(error_msg)
        await msg.update()


@cl.action_callback("session_info")
async def on_session_info(action):
    """Show current session information."""
    session_id = cl.user_session.get("rag_session_id")
    if not session_id or session_id not in rag_instances:
        await cl.Message(content="❌ **No active session found.**").send()
        return
    
    rag = rag_instances[session_id]
    info = rag.get_session_info()
    
    info_text = f"""📊 **Session Information**

**Session ID:** `{info['session_id']}`
**Topic:** {info.get('topic', 'Not determined yet')}
**Questions Asked:** {info['total_questions']}
**Articles Indexed:** {info['total_articles']}
**Search Terms Used:** {info['total_search_terms']}
**Created:** {info['created_at'][:19]}Z

This session has indexed {info['total_articles']} unique PubMed articles to answer your questions.
"""
    
    await cl.Message(content=info_text).send()


@cl.action_callback("new_session")
async def on_new_session(action):
    """Create a new session """
    
    # Clean up old session
    old_session_id = cl.user_session.get("rag_session_id")
    if old_session_id and old_session_id in rag_instances:
        try:
            rag_instances[old_session_id].close()
            del rag_instances[old_session_id]
        except Exception as e:
            logger.warning(f"Error cleaning up old session: {e}")
    
    # Start new session (reuse the start function logic)
    await start()


@cl.on_chat_end
async def on_chat_end():
    """Clean up when chat ends."""
    session_id = cl.user_session.get("rag_session_id")
    if session_id and session_id in rag_instances:
        try:
            rag_instances[session_id].close()
            del rag_instances[session_id]
            logger.info(f"Cleaned up session: {session_id}")
        except Exception as e:
            logger.warning(f"Error cleaning up session: {e}")

if __name__ == "__main__":
    # This allows running the app directly
    import subprocess
    import sys
    
    try:
        subprocess.run([sys.executable, "-m", "chainlit", "run", __file__], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Chainlit: {e}")
    except KeyboardInterrupt:
        print("\nShutting down...") 