"""
Utility functions for ADHD Memory Assistant
"""
from typing import Dict, List, Any
import json
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

def run_async(coroutine):
    """Helper function to run async code in Streamlit."""
    with ThreadPoolExecutor() as executor:
        future = asyncio.run_coroutine_threadsafe(coroutine, asyncio.get_event_loop())
        return future.result()

def initialize_session_state(st) -> None:
    """Initialize Streamlit session state variables."""
    if 'current_task' not in st.session_state:
        st.session_state.current_task = None
    if 'prompts' not in st.session_state:
        st.session_state.prompts = []
    if 'prompt_types' not in st.session_state:
        st.session_state.prompt_types = {}
    if 'responses' not in st.session_state:
        st.session_state.responses = {}
    if 'effectiveness_ratings' not in st.session_state:
        st.session_state.effectiveness_ratings = {}
    if 'context' not in st.session_state:
        st.session_state.context = None
    if 'similar_tasks' not in st.session_state:
        st.session_state.similar_tasks = []

def format_timestamp(timestamp: str) -> str:
    """Format ISO timestamp to readable string."""
    dt = datetime.fromisoformat(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def calculate_average_rating(ratings: Dict[str, int]) -> float:
    """Calculate average rating from effectiveness ratings."""
    if not ratings:
        return 0.0
    return sum(ratings.values()) / len(ratings)

def extract_memory_triggers(text: str) -> List[str]:
    """Extract potential memory triggers from text."""
    trigger_phrases = [
        "reminds me of",
        "makes me think of",
        "similar to",
        "connects to",
        "related to",
        "like when",
        "remember when"
    ]
    
    triggers = []
    for phrase in trigger_phrases:
        if phrase in text.lower():
            # Split on the phrase and take the latter part
            parts = text.lower().split(phrase)
            for part in parts[1:]:  # Skip the first part (before the phrase)
                # Clean up and add the trigger
                trigger = part.split('.')[0].strip()
                if trigger:
                    triggers.append(trigger)
    
    return triggers

def sanitize_prompt_type(prompt_type: str) -> str:
    """Sanitize and validate prompt type."""
    valid_types = {'location', 'time', 'activity', 'sensory'}
    prompt_type = prompt_type.lower().strip()
    return prompt_type if prompt_type in valid_types else 'other'

def create_memory_metadata(task: str, 
                         effectiveness_ratings: Dict[str, int], 
                         prompt_types: Dict[str, str]) -> Dict[str, Any]:
    """Create metadata for memory storage."""
    return {
        'timestamp': datetime.now().isoformat(),
        'task': task,
        'average_effectiveness': calculate_average_rating(effectiveness_ratings),
        'prompt_types': json.dumps(prompt_types),
        'total_prompts': len(effectiveness_ratings)
    }