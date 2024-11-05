"""
Main Streamlit application for ADHD Memory Assistant
"""
import streamlit as st
import anthropic
from datetime import datetime
from typing import Dict
import os
from dotenv import load_dotenv

from prompt_engine import PromptEngine
from utils import initialize_session_state, run_async, calculate_average_rating

# Load environment variables
load_dotenv()

# Initialize Anthropic client
client = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY"))

def main():
    st.title("ADHD Memory Assistant")
    initialize_session_state(st)

    # Ensure PromptEngine is initialized
    if 'engine' not in st.session_state:
        st.session_state.engine = PromptEngine(client)

    # Task Input Section
    with st.container():
        st.subheader("Task Description")
        task = st.text_area(
            "What task do you need help remembering?",
            help="Describe what you're trying to remember. Include any relevant details."
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            generate_button = st.button("Generate Memory Prompts")
        with col2:
            if st.session_state.similar_tasks:
                st.info(f"{len(st.session_state.similar_tasks)} similar memories found")

        if generate_button and task:
            with st.spinner("Analyzing and generating memory prompts..."):
                try:
                    # Get context and generate prompts
                    st.session_state.context = st.session_state.engine.get_context_for_task(task)
                    st.session_state.similar_tasks = st.session_state.context.get('similar_tasks', [])
                    
                    prompts, prompt_types = run_async(
                        st.session_state.engine.generate_prompts(task, st.session_state.context)
                    )
                    
                    # Update session state
                    st.session_state.current_task = task
                    st.session_state.prompts = prompts
                    st.session_state.prompt_types = prompt_types
                    st.session_state.responses = {}
                    st.session_state.effectiveness_ratings = {}
                    
                    if 'error' in st.session_state:
                        del st.session_state.error
                        
                except Exception as e:
                    st.session_state.error = str(e)

    # Error Display
    if 'error' in st.session_state:
        st.error(st.session_state.error)
        if st.button("Clear Error"):
            del st.session_state.error

    # Similar Tasks Display
    if st.session_state.similar_tasks:
        with st.expander("View Similar Memory Sessions"):
            for task_data in st.session_state.similar_tasks:
                st.write(f"Previous task: {task_data['task']}")
                st.write(f"Effectiveness: {task_data['effectiveness']:.1f}/5")
                st.divider()

    # Prompts and Responses Section
    if st.session_state.current_task and st.session_state.prompts:
        with st.container():
            st.subheader("Memory Assistance Questions")
            
            # Group and display prompts by type
            grouped_prompts = {}
            for prompt in st.session_state.prompts:
                prompt_type = st.session_state.prompt_types.get(prompt, 'other')
                if prompt_type not in grouped_prompts:
                    grouped_prompts[prompt_type] = []
                grouped_prompts[prompt_type].append(prompt)

            for prompt_type, prompts in grouped_prompts.items():
                with st.expander(f"{prompt_type.title()} Questions", expanded=True):
                    for i, prompt in enumerate(prompts):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            response = st.text_area(
                                f"Q{i+1}: {prompt}",
                                key=f"response_{prompt_type}_{i}",
                                help="Your answer will help build memory associations"
                            )
                            if response:
                                st.session_state.responses[prompt] = response
                        
                        with col2:
                            effectiveness = st.select_slider(
                                "How helpful was this question?",
                                options=[1, 2, 3, 4, 5],
                                value=3,
                                key=f"effectiveness_{prompt_type}_{i}"
                            )
                            st.session_state.effectiveness_ratings[prompt] = effectiveness

        # Progress Indicator
        if st.session_state.responses:
            progress = len(st.session_state.responses) / len(st.session_state.prompts)
            st.progress(progress, text=f"Progress: {int(progress * 100)}%")

        # Save Responses
        if st.button("Save Responses", 
                    help="Save your responses to improve future memory retrieval",
                    disabled=len(st.session_state.responses) == 0):
            with st.spinner("Saving responses and updating memory patterns..."):
                try:
                    st.session_state.engine.update_memory_patterns(
                        st.session_state.current_task,
                        st.session_state.prompts,
                        st.session_state.responses,
                        st.session_state.effectiveness_ratings
                    )
                    
                    avg_effectiveness = calculate_average_rating(
                        st.session_state.effectiveness_ratings
                    )
                    st.success("Responses saved successfully!")
                    st.info(f"Average prompt effectiveness: {avg_effectiveness:.1f}/5")
                    
                except Exception as e:
                    st.error(f"Error saving responses: {str(e)}")

if __name__ == "__main__":
    main()