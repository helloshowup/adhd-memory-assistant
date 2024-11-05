"""
Prompt generation and template management for ADHD Memory Assistant
"""
from typing import Dict, List, Tuple
import json
from datetime import datetime
import anthropic
from memory_manager import EnhancedMemoryManager
from utils import extract_memory_triggers

class PromptTemplate:
    """ADHD-specific prompt template manager"""
    
    def __init__(self):
        self.template = {
            "anchors": {
                "location": [
                    "Where were you when you first thought about this?",
                    "Which places are associated with this?",
                    "Where do you need to be to work on this?",
                    "Are there any specific locations that remind you of this?"
                ],
                "time": [
                    "What time of day did you think of this?",
                    "Are there specific deadlines or time constraints?",
                    "What activities usually happen around the same time?",
                    "How does this fit into your daily routine?"
                ],
                "activity": [
                    "What were you doing when you thought of this?",
                    "Which regular activities does this relate to?",
                    "What needs to happen before/after this?",
                    "Are there any similar tasks you've done before?"
                ],
                "sensory": [
                    "What sounds or sights remind you of this?",
                    "Are there any distinct physical objects related to this?",
                    "What environmental cues trigger thoughts about this?",
                    "Can you identify any specific feelings associated with this?"
                ]
            },
            "priority_weights": {
                "urgency": 1.5,
                "importance": 1.3,
                "complexity": 1.2
            }
        }
    
    def analyze_task(self, task: str) -> dict:
        """Analyze task characteristics to determine prompt selection weights"""
        analysis = {
            "urgency": 1.0,
            "importance": 1.0,
            "complexity": 1.0
        }
        
        # Basic keyword analysis
        if any(word in task.lower() for word in ["urgent", "asap", "tomorrow", "soon", "deadline"]):
            analysis["urgency"] = 1.5
        if any(word in task.lower() for word in ["important", "critical", "essential", "key"]):
            analysis["importance"] = 1.3
        if any(word in task.lower() for word in ["complex", "difficult", "multiple", "steps"]):
            analysis["complexity"] = 1.2
            
        return analysis
    
    def select_prompts(self, task: str) -> Tuple[List[str], Dict[str, str]]:
        """Select appropriate prompts based on task analysis"""
        analysis = self.analyze_task(task)
        selected_prompts = []
        prompt_types = {}
        
        for anchor_type, prompts in self.template["anchors"].items():
            # Calculate number of prompts needed for this type
            num_prompts = 1
            if anchor_type == "time" and analysis["urgency"] > 1.0:
                num_prompts += 1
            if anchor_type == "activity" and analysis["complexity"] > 1.0:
                num_prompts += 1
                
            # Select prompts and track their types
            import random
            selected = random.sample(prompts, min(num_prompts, len(prompts)))
            selected_prompts.extend(selected)
            
            for prompt in selected:
                prompt_types[prompt] = anchor_type
            
        return selected_prompts, prompt_types


class PromptEngine:
    """Main prompt engine with enhanced pattern tracking"""
    
    def __init__(self, client: anthropic.Client):
        self.client = client
        self.max_retries = 3
        self.retry_delay = 2
        self.memory_manager = EnhancedMemoryManager()
        self.template_manager = PromptTemplate()
        self.current_memory_triggers = {}

    async def generate_prompts(self, task: str, context: Dict = None) -> Tuple[List[str], Dict[str, str]]:
        """Generate prompts using memory patterns and error recovery"""
        try:
            memory_patterns = self.memory_manager.retrieve_memory_patterns(task)
            base_prompts, prompt_types = self.template_manager.select_prompts(task)

            enhanced_prompts = await self._enhance_prompts_with_patterns(
                task, base_prompts, memory_patterns
            )

            return enhanced_prompts, prompt_types

        except Exception as e:
            print(f"Error in prompt generation: {str(e)}")
            return base_prompts, prompt_types

    async def _enhance_prompts_with_patterns(
        self, task: str, base_prompts: List[str], 
        memory_patterns: Dict) -> List[str]:
        """Enhance prompts using successful memory patterns"""
        
        system_prompt = """
        You are a specialized assistant helping users with ADHD improve memory retrieval.
        Enhance these prompts using:
        1. Proven memory triggers
        2. Successful memory chains
        3. Effective prompt sequences
        4. Conversational flow
        
        Return prompts as JSON with:
        - question
        - trigger_type (visual, spatial, temporal, procedural, emotional)
        - connection
        """

        try:
            message = await self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1500,
                messages=[{
                    "role": "user",
                    "content": f"""
                    Task: {task}
                    Base Prompts: {json.dumps(base_prompts)}
                    Effective Patterns: {json.dumps(memory_patterns['effective_patterns'])}
                    Suggested Sequence: {json.dumps(memory_patterns['suggested_sequence'])}
                    """
                }],
                system=system_prompt
            )

            enhanced_data = json.loads(message.content[0].text)
            
            # Update trigger tracking
            for prompt_data in enhanced_data:
                self.current_memory_triggers[prompt_data['question']] = {
                    'trigger_type': prompt_data['trigger_type'],
                    'expected_connection': prompt_data['connection']
                }
            
            return [p['question'] for p in enhanced_data]

        except Exception:
            return base_prompts

    def update_memory_patterns(self, task: str, prompts: List[str], 
                             responses: Dict[str, str], 
                             effectiveness_ratings: Dict[str, int]) -> None:
        """Update memory patterns with session results"""
        prompt_responses = [
            {
                'prompt': prompt,
                'response': responses.get(prompt, ''),
                'type': self.current_memory_triggers.get(prompt, {}).get('trigger_type', 'unknown'),
                'trigger_data': self.current_memory_triggers.get(prompt, {})
            }
            for prompt in prompts if prompt in responses
        ]

        memory_triggers = {
            prompt: extract_memory_triggers(response)
            for prompt, response in responses.items()
        }

        self.memory_manager.store_memory_pattern(
            task=task,
            prompt_responses=prompt_responses,
            effectiveness_ratings=effectiveness_ratings,
            memory_triggers=memory_triggers
        )

        self.current_memory_triggers = {}