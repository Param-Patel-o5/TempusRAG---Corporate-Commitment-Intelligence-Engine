#!/usr/bin/env python3
"""
TempusRAG Streamlit Web Application

A comprehensive web interface for analyzing corporate credibility through 
SEC filings analysis. Features promise extraction, cross-year reasoning,
and natural language querying capabilities.
"""

import re
import time
import logging
from typing import Optional, Tuple, List, Dict, Any

import streamlit as st

# TempusRAG imports
from src.pipeline import run_tempusrag_pipeline
from src.query import process_query
from src.models import CompanyCredibilityReport
from src.config import GEMINI_API_KEY, GROQ_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="TempusRAG - Corporate Credibility Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "TempusRAG analyzes corporate credibility through SEC filings"
    }
)

def init_session_state() -> None:
    """Initialize all session state variables with defaults."""
    defaults = {
        'ticker': '',
        'report': None,
        'conversation': [],
        'gemini_override_key': '',
        'groq_override_key': '',
        'use_custom_gemini': False,
        'use_custom_groq': False,
        'analysis_running': False,
        'last_analysis_time': None,
        'show_api_form': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def render_custom_css() -> None:
    """Inject custom CSS for glass morphism and modern styling."""
    st.markdown("""
    <style>
    /* Global theme overrides */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Glass morphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 24px;
        margin: 16px 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    
    /* Hero score display */
    .hero-score {
        font-size: 72px;
        font-weight: bold;
        background: linear-gradient(45deg, #00d4aa, #8a2be2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 0;
        line-height: 1.2;
    }
    
    /* Gradient buttons */
    .gradient-button {
        background: linear-gradient(45deg, #00d4aa, #00a8cc);
        border: none;
        border-radius: 8px;
        color: white;
        padding: 12px 24px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .gradient-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 212, 170, 0.4);
    }
    
    /* Status badges */
    .status-delivered {
        background: #10b981;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
    }
    
    .status-partial {
        background: #f59e0b;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
    }
    
    .status-abandoned {
        background: #ef4444;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
    }
    
    /* Domain badges */
    .domain-technology {
        background: #3b82f6;
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 500;
    }
    
    .domain-claims {
        background: #8b5cf6;
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 500;
    }
    
    .domain-growth {
        background: #10b981;
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 500;
    }
    
    .domain-finance {
        background: #f59e0b;
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 500;
    }
    
    .domain-operations {
        background: #6b7280;
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 500;
    }
    
    /* Chat styling */
    .chat-message {
        padding: 12px;
        margin: 8px 0;
        border-radius: 12px;
        max-width: 85%;
    }
    
    .chat-user {
        background: linear-gradient(45deg, #00d4aa, #00a8cc);
        color: white;
        margin-left: auto;
        text-align: right;
    }
    
    .chat-assistant {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
    }
    
    /* Progress bar styling */
    .domain-progress {
        height: 8px;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 4px;
        overflow: hidden;
        margin: 4px 0;
    }
    
    .domain-progress-fill {
        height: 100%;
        background: linear-gradient(45deg, #00d4aa, #8a2be2);
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    
    /* Red flag styling */
    .red-flag {
        background: rgba(239, 68, 68, 0.1);
        border-left: 4px solid #ef4444;
        padding: 12px;
        margin: 8px 0;
        border-radius: 4px;
        color: #fecaca;
    }
    
    /* Metric cards */
    .metric-card {
        text-align: center;
        padding: 20px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .metric-number {
        font-size: 36px;
        font-weight: bold;
        margin: 8px 0;
    }
    
    .metric-label {
        font-size: 14px;
        opacity: 0.8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    </style>
    """, unsafe_allow_html=True)
def render_header() -> None:
    """Render the main application header with logo and navigation."""
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 3rem; font-weight: bold; margin-bottom: 0.5rem; 
                   background: linear-gradient(45deg, #00d4aa, #8a2be2);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            📊 TempusRAG
        </h1>
        <p style="font-size: 1.2rem; opacity: 0.8; margin-bottom: 2rem;">
            Corporate Credibility Analysis through SEC Filings
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar_api_info() -> None:
    """Display current API key configuration and status in sidebar."""
    st.sidebar.markdown("### 🔑 API Configuration")
    
    gemini_key, groq_key = get_active_api_keys()
    
    # Gemini status
    if gemini_key:
        if st.session_state.get('use_custom_gemini'):
            st.sidebar.success("✅ Gemini: Custom key")
        else:
            st.sidebar.success("✅ Gemini: App key")
    else:
        st.sidebar.error("❌ Gemini: No key")
    
    # Groq status  
    if groq_key:
        if st.session_state.get('use_custom_groq'):
            st.sidebar.success("✅ Groq: Custom key")
        else:
            st.sidebar.success("✅ Groq: App key")
    else:
        st.sidebar.error("❌ Groq: No key")
    
    # API key override form
    if st.sidebar.button("⚙️ Use my own API keys", type="secondary"):
        st.session_state.show_api_form = not st.session_state.show_api_form
    
    if st.session_state.show_api_form:
        render_api_key_form()

def render_api_key_form() -> None:
    """Render the API key override form in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Custom API Keys")
    st.sidebar.markdown("*Keys are session-only, not stored*")
    
    # Gemini key form
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        use_gemini = st.checkbox("Use Gemini key", value=st.session_state.use_custom_gemini)
    with col2:
        if use_gemini:
            st.session_state.use_custom_gemini = True
    
    if use_gemini:
        gemini_key = st.sidebar.text_input(
            "Gemini API Key",
            type="password",
            value=st.session_state.gemini_override_key,
            help="Get from https://aistudio.google.com/app/apikey"
        )
        if gemini_key != st.session_state.gemini_override_key:
            st.session_state.gemini_override_key = gemini_key
    else:
        st.session_state.use_custom_gemini = False
        st.session_state.gemini_override_key = ""
    
    # Groq key form
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        use_groq = st.checkbox("Use Groq key", value=st.session_state.use_custom_groq)
    with col2:
        if use_groq:
            st.session_state.use_custom_groq = True
    
    if use_groq:
        groq_key = st.sidebar.text_input(
            "Groq API Key",
            type="password", 
            value=st.session_state.groq_override_key,
            help="Get from https://console.groq.com/keys"
        )
        if groq_key != st.session_state.groq_override_key:
            st.session_state.groq_override_key = groq_key
    else:
        st.session_state.use_custom_groq = False
        st.session_state.groq_override_key = ""

def render_sidebar_requirements() -> None:
    """Display API requirements and setup instructions."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Requirements")
    st.sidebar.markdown("""
    **Required APIs:**
    - **Gemini**: Promise extraction from SEC filings
    - **Groq**: Cross-year reasoning and judgment
    
    **Setup Options:**
    1. Use app's built-in keys (limited usage)
    2. Provide your own keys above (unlimited)
    
    **Getting API Keys:**
    - [Gemini API Key](https://aistudio.google.com/app/apikey) (Free tier: 20 requests/day)
    - [Groq API Key](https://console.groq.com/keys) (Free tier: 70K tokens/minute)
    """)

def get_active_api_keys() -> Tuple[str, str]:
    """
    Returns (gemini_key, groq_key) based on current configuration.
    Prioritizes custom user keys over app configuration.
    """
    # Determine Gemini key
    if st.session_state.get('use_custom_gemini') and st.session_state.get('gemini_override_key'):
        gemini_key = st.session_state.gemini_override_key
        logger.debug("Using custom Gemini key")
    else:
        gemini_key = GEMINI_API_KEY
        logger.debug("Using app Gemini key")
    
    # Determine Groq key  
    if st.session_state.get('use_custom_groq') and st.session_state.get('groq_override_key'):
        groq_key = st.session_state.groq_override_key
        logger.debug("Using custom Groq key")
    else:
        groq_key = GROQ_API_KEY
        logger.debug("Using app Groq key")
    
    return gemini_key, groq_key

def render_hero_card(report: CompanyCredibilityReport) -> None:
    """Render the main hero card with credibility score and summary."""
    delivered_percent = (report.delivered / report.total_promises * 100) if report.total_promises > 0 else 0
    
    st.markdown(f"""
    <div class="glass-card">
        <div class="hero-score">{report.overall_score:.0f}</div>
        <p style="text-align: center; font-size: 1.2rem; margin-top: 1rem; opacity: 0.9;">
            <strong>{report.company}</strong> has delivered on <strong>{report.delivered}</strong> of 
            <strong>{report.total_promises}</strong> forward-looking promises across 
            <strong>{len(report.years_analyzed)}</strong> fiscal years.
            Overall delivery rate: <strong>{delivered_percent:.1f}%</strong>. 
            <strong>{len(report.red_flags)}</strong> red flag(s) detected.
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_ticker_input() -> None:
    """Render the ticker input section with validation and analysis trigger."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        ticker = st.text_input(
            "Enter company ticker",
            value=st.session_state.ticker,
            placeholder="e.g., HIG, NVDA, AAPL...",
            help="Enter a valid stock ticker symbol",
            key="ticker_input"
        )
        if ticker != st.session_state.ticker:
            st.session_state.ticker = ticker.upper()
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Align with input
        analyze_button = st.button(
            "🔍 Analyze",
            type="primary",
            disabled=st.session_state.analysis_running
        )
    
    st.markdown("""
    <p style="margin-top: 1rem; opacity: 0.7; font-size: 0.9rem;">
        💡 Press Enter or click Analyze to start. First analysis may take 1-2 minutes.
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle analysis trigger
    if analyze_button or st.session_state.get('trigger_analysis'):
        if st.session_state.get('trigger_analysis'):
            st.session_state.trigger_analysis = False
        
        # Validate ticker
        if not ticker or not re.match(r'^[A-Z]{1,5}$', ticker):
            st.error("Please enter a valid ticker (1-5 letters, e.g., AAPL, HIG)")
            return
        
        # Get and validate API keys
        gemini_key, groq_key = get_active_api_keys()
        
        if not gemini_key:
            st.error("""
            **Gemini API key required** for promise extraction. 
            
            **Options:**
            1. Set `GEMINI_API_KEY` in your `.env` file
            2. Use custom key via Settings in sidebar
            
            Get your free key at: https://aistudio.google.com/app/apikey
            """)
            return
            
        if not groq_key:
            st.error("""
            **Groq API key required** for reasoning and judgment.
            
            **Options:**
            1. Set `GROQ_API_KEY` in your `.env` file  
            2. Use custom key via Settings in sidebar
            
            Get your free key at: https://console.groq.com/keys
            """)
            return
        
        # Run analysis
        run_analysis(ticker, gemini_key, groq_key)

def run_analysis(ticker: str, gemini_key: str, groq_key: str) -> None:
    """Execute the TempusRAG analysis pipeline with progress tracking."""
    st.session_state.analysis_running = True
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    stages = [
        "🔍 Ingesting SEC filings...",
        "✂️ Chunking documents...", 
        "🧠 Generating embeddings...",
        "📝 Extracting promises...",
        "⚖️ Cross-year reasoning...",
        "📊 Generating report..."
    ]
    
    try:
        # Update progress through stages
        for i, stage in enumerate(stages):
            progress_bar.progress((i + 1) / len(stages))
            status_text.text(f"Analyzing {ticker}... {stage}")
            
            if i < len(stages) - 1:  # Don't sleep on last stage
                time.sleep(0.5)  # Brief delay for UX
        
        # Run the actual pipeline
        status_text.text(f"Running TempusRAG pipeline for {ticker}...")
        
        report = run_tempusrag_pipeline(
            ticker=ticker,
            company_display_name=None,  # Will use ticker as fallback
            gemini_api_key=gemini_key,
            groq_api_key=groq_key,
            force_reingest=False
        )
        
        # Success
        progress_bar.progress(1.0)
        status_text.text("✅ Analysis complete!")
        time.sleep(1)
        
        # Store results
        st.session_state.report = report
        st.session_state.last_analysis_time = time.time()
        st.session_state.conversation = []  # Reset chat
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"Analysis complete for {report.company}! 🎉")
        st.rerun()  # Refresh to show results
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        
        error_msg = str(e).lower()
        
        # Handle specific error types
        if '401' in error_msg or 'unauthorized' in error_msg:
            st.error("❌ **Invalid API key** - Check your credentials and try again.")
        elif '429' in error_msg or 'rate limit' in error_msg or 'quota' in error_msg:
            st.error("⏳ **API rate limit exceeded** - Wait a few minutes or use different API keys.")
        elif 'network' in error_msg or 'connection' in error_msg:
            st.error("🌐 **Connection error** - Check your internet connection and try again.")
        elif 'not found' in error_msg and 'ticker' in error_msg:
            st.error(f"❌ **Ticker '{ticker}' not found** - Please check the ticker symbol.")
        else:
            st.error(f"❌ **Analysis failed**: {str(e)}")
            
        logger.error(f"Pipeline failed for {ticker}: {e}")
        
    finally:
        st.session_state.analysis_running = False
def render_metrics(report: CompanyCredibilityReport) -> None:
    """Render three metric cards showing delivery statistics."""
    st.markdown("### 📈 Key Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Delivered</div>
            <div class="metric-number" style="color: #10b981;">{report.delivered}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Abandoned</div>
            <div class="metric-number" style="color: #ef4444;">{report.abandoned}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Pending</div>
            <div class="metric-number" style="color: #f59e0b;">{report.partial}</div>
        </div>
        """, unsafe_allow_html=True)

def render_domain_scores(report: CompanyCredibilityReport) -> None:
    """Render domain scores with horizontal progress bars."""
    if not report.domain_scores:
        return
        
    st.markdown("### 🎯 Domain Performance")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    # Filter out zero scores
    non_zero_domains = {k: v for k, v in report.domain_scores.items() if v > 0}
    
    if not non_zero_domains:
        st.markdown("*No domain scores available*")
    else:
        for domain, score in non_zero_domains.items():
            col1, col2, col3 = st.columns([2, 6, 1])
            
            with col1:
                st.markdown(f"**{domain}**")
            
            with col2:
                # Custom progress bar
                st.markdown(f"""
                <div class="domain-progress">
                    <div class="domain-progress-fill" style="width: {score}%"></div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"**{score:.0f}**")
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_red_flags(report: CompanyCredibilityReport) -> None:
    """Render red flags section if any exist."""
    if not report.red_flags:
        return
        
    st.markdown(f"### 🚩 Red Flags ({len(report.red_flags)})")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    for flag in report.red_flags:
        st.markdown(f'<div class="red-flag">{flag}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def get_domain_badge_class(domain: str) -> str:
    """Get CSS class for domain badge based on domain name."""
    domain_lower = domain.lower()
    if 'technology' in domain_lower:
        return 'domain-technology'
    elif 'claims' in domain_lower:
        return 'domain-claims'
    elif 'growth' in domain_lower:
        return 'domain-growth'
    elif 'finance' in domain_lower:
        return 'domain-finance'
    elif 'operations' in domain_lower:
        return 'domain-operations'
    else:
        return 'domain-operations'  # Default

def get_status_badge_class(status: str) -> str:
    """Get CSS class for status badge based on delivery status."""
    status_lower = status.lower()
    if 'delivered' in status_lower:
        return 'status-delivered'
    elif 'partial' in status_lower:
        return 'status-partial'
    else:
        return 'status-abandoned'
def render_promises_table(report: CompanyCredibilityReport) -> None:
    """Render promises in tabbed interface with filtering options."""
    if not hasattr(report, '_promises_by_year') and hasattr(report, 'delivery_evidence'):
        # Extract promises from delivery evidence if available
        promises = []
        if report.delivery_evidence:
            for evidence in report.delivery_evidence:
                promises.append({
                    'text': evidence.promise.promise_text,
                    'domain': evidence.promise.domain,
                    'status': evidence.status,
                    'year': evidence.promise.year_made,
                    'confidence': getattr(evidence.promise, 'confidence_score', 0.5)
                })
    else:
        # Fallback if no detailed promise data
        promises = []
    
    st.markdown("### 📋 Promises Analysis")
    
    # Tab counts
    delivered_count = sum(1 for p in promises if 'delivered' in p.get('status', '').lower())
    at_risk_count = len(promises) - delivered_count
    
    tab1, tab2, tab3 = st.tabs([
        f"All ({len(promises)})",
        f"Delivered ({delivered_count})", 
        f"At Risk ({at_risk_count})"
    ])
    
    with tab1:
        render_promises_list(promises, "all")
    
    with tab2:
        delivered_promises = [p for p in promises if 'delivered' in p.get('status', '').lower()]
        render_promises_list(delivered_promises, "delivered")
    
    with tab3:
        at_risk_promises = [p for p in promises if 'delivered' not in p.get('status', '').lower()]
        render_promises_list(at_risk_promises, "at_risk")

def render_promises_list(promises: List[Dict], filter_type: str) -> None:
    """Render a list of promises with expandable details."""
    if not promises:
        st.markdown("*No promises in this category*")
        return
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    for i, promise in enumerate(promises):
        # Truncate text for display
        display_text = promise['text'][:60] + "..." if len(promise['text']) > 60 else promise['text']
        
        # Create expandable section
        with st.expander(f"{display_text}"):
            col1, col2, col3 = st.columns([6, 2, 2])
            
            with col1:
                st.markdown(f"**Full Promise:** {promise['text']}")
            
            with col2:
                domain_class = get_domain_badge_class(promise['domain'])
                st.markdown(f'<span class="{domain_class}">{promise["domain"]}</span>', 
                           unsafe_allow_html=True)
            
            with col3:
                status_class = get_status_badge_class(promise['status'])
                st.markdown(f'<span class="{status_class}">{promise["status"]}</span>', 
                           unsafe_allow_html=True)
            
            # Additional details
            st.markdown(f"**Year Made:** {promise.get('year', 'Unknown')}")
            st.markdown(f"**Confidence Score:** {promise.get('confidence', 0.5):.2f}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_chat_interface() -> None:
    """Render the chat interface for asking questions about the company."""
    st.markdown("### 💬 Ask about this company")
    
    if not st.session_state.report:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("*Run an analysis first to ask questions about the company*")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Chat container
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    # Display conversation history
    chat_container = st.container()
    
    with chat_container:
        if st.session_state.conversation:
            for message in st.session_state.conversation:
                if message['role'] == 'user':
                    st.markdown(f"""
                    <div class="chat-message chat-user">
                        {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message chat-assistant">
                        {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show sources if available
                    if 'sources' in message:
                        sources_text = " | ".join([f"[{s}]" for s in message['sources']])
                        st.markdown(f"**Sources:** {sources_text}")
        else:
            st.markdown("*Start a conversation by asking a question below*")
    
    # Input area
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_question = st.text_input(
            "Ask a question",
            placeholder="e.g., What technology investments did they promise?",
            key="chat_input"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Align with input
        send_button = st.button("Send", type="primary")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle question submission
    if send_button and user_question.strip():
        handle_chat_question(user_question.strip())
    elif st.session_state.get('submit_question'):
        # Handle Enter key submission
        question = st.session_state.submit_question
        st.session_state.submit_question = ""
        handle_chat_question(question)

def handle_chat_question(question: str) -> None:
    """Process a chat question and generate response."""
    if not st.session_state.report:
        st.error("Please run an analysis first")
        return
    
    # Add user message to conversation
    st.session_state.conversation.append({
        'role': 'user',
        'content': question
    })
    
    try:
        # Get API keys for query processing
        gemini_key, groq_key = get_active_api_keys()
        
        # Process the query
        with st.spinner("Thinking..."):
            result = process_query(
                query=question,
                ticker=st.session_state.ticker,
                conversation_history=st.session_state.conversation[:-1],  # Exclude current question
                api_key=gemini_key  # Query system uses Gemini
            )
        
        # Extract answer and sources
        answer = result.get('answer', 'No answer generated')
        source_chunks = result.get('source_chunks', [])
        
        # Format sources
        sources = []
        if source_chunks:
            sources = [f"Chunk_{i+1}" for i in range(len(source_chunks))]
        
        # Add assistant response
        assistant_message = {
            'role': 'assistant',
            'content': answer
        }
        if sources:
            assistant_message['sources'] = sources
        
        st.session_state.conversation.append(assistant_message)
        
        # Clear input and refresh
        st.session_state.chat_input = ""
        st.rerun()
        
    except Exception as e:
        st.error(f"Failed to generate response: {str(e)}")
        logger.error(f"Chat query failed: {e}")

def render_footer() -> None:
    """Render the application footer with attribution."""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; opacity: 0.7;">
        <p style="margin: 0; font-size: 0.9rem;">
            Real SEC filings analyzed via Claude AI | Powered by Groq LLM reasoning | 
            Last updated: {timestamp}
        </p>
    </div>
    """.format(
        timestamp=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    ), unsafe_allow_html=True)

def render_empty_state() -> None:
    """Render the initial empty state when no analysis has been run."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("""
    ## 🚀 Get Started
    
    Enter a company ticker above to begin comprehensive credibility analysis.
    
    **What TempusRAG does:**
    1. **Ingests SEC filings** from recent years
    2. **Extracts forward-looking promises** using AI
    3. **Judges delivery** by analyzing subsequent filings  
    4. **Generates credibility scores** across business domains
    5. **Enables natural language queries** about the company
    
    **Example tickers to try:** HIG, NVDA, AAPL, TSLA, MSFT
    
    *First analysis takes 1-2 minutes while we process the filings*
    """)
    st.markdown('</div>', unsafe_allow_html=True)

@st.cache_resource
def load_cik_map():
    """Cache the CIK mapping to avoid repeated API calls."""
    try:
        from src.ingestion import load_ticker_cik_map
        return load_ticker_cik_map()
    except Exception as e:
        logger.error(f"Failed to load CIK map: {e}")
        return {}

@st.cache_data(ttl=3600)  # Cache for 1 hour
def cache_pipeline_result(ticker: str, timestamp: int):
    """Cache pipeline results to avoid re-analysis."""
    # This is a placeholder - actual caching would store the report
    # For now, we rely on session state
    pass
def main() -> None:
    """Main application entry point."""
    # Initialize session state
    init_session_state()
    
    # Inject custom CSS
    render_custom_css()
    
    # Render header
    render_header()
    
    # Sidebar content
    with st.sidebar:
        render_sidebar_api_info()
        render_sidebar_requirements()
    
    # Main content area
    if st.session_state.report is None:
        # No analysis yet - show input and instructions
        render_ticker_input()
        render_empty_state()
    else:
        # Analysis complete - show results
        report = st.session_state.report
        
        # Hero card with main score
        render_hero_card(report)
        
        # Two-column layout for detailed results
        left_col, right_col = st.columns([2, 1])
        
        with left_col:
            # Left column: Metrics and analysis
            render_metrics(report)
            render_domain_scores(report)
            render_red_flags(report)
            render_promises_table(report)
        
        with right_col:
            # Right column: Chat interface
            render_chat_interface()
    
    # Footer
    render_footer()

# Handle Enter key submission for ticker input
if 'ticker_input' in st.session_state and st.session_state.ticker_input:
    if st.session_state.ticker_input != st.session_state.ticker:
        st.session_state.ticker = st.session_state.ticker_input.upper()
        # Trigger analysis on Enter if ticker is valid
        if re.match(r'^[A-Z]{1,5}$', st.session_state.ticker):
            st.session_state.trigger_analysis = True
            st.rerun()

# Application entry point
if __name__ == "__main__":
    main()