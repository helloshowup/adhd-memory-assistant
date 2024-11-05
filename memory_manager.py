"""
Memory management and pattern tracking for ADHD Memory Assistant
"""
from typing import Dict, List, Tuple
import json
from datetime import datetime
import sys
import pysqlite3
sys.modules['sqlite3'] = pysqlite3
import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class PromptMemory:
    """Manages prompt effectiveness history and pattern matching"""
    
    def __init__(self):
        self.success_threshold = 4  # Rating >= 4 is considered successful
        self.similarity_threshold = 0.6  # Minimum similarity score
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings using TF-IDF"""
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            return 0.0

    def find_similar_tasks(self, current_task: str, task_history: List[Dict]) -> List[Dict]:
        """Find similar tasks from history based on text similarity"""
        similar_tasks = []
        
        for hist_task in task_history:
            similarity = self.calculate_similarity(current_task, hist_task['task'])
            if similarity >= self.similarity_threshold:
                similar_tasks.append({
                    'task': hist_task['task'],
                    'similarity': similarity,
                    'prompts': hist_task['prompts'],
                    'effectiveness': hist_task['effectiveness']
                })
        
        return sorted(similar_tasks, key=lambda x: x['similarity'], reverse=True)

    def extract_successful_patterns(self, task_history: List[Dict]) -> Dict[str, List[str]]:
        """Extract patterns from successful prompts"""
        successful_patterns = {
            'location': [],
            'time': [],
            'activity': [],
            'sensory': []
        }
        
        for task in task_history:
            for prompt_data in task['prompts']:
                if prompt_data['effectiveness'] >= self.success_threshold:
                    for anchor_type in successful_patterns.keys():
                        if anchor_type in prompt_data['type']:
                            successful_patterns[anchor_type].append(prompt_data['prompt'])
        
        # Remove duplicates while preserving order
        for anchor_type in successful_patterns:
            successful_patterns[anchor_type] = list(dict.fromkeys(successful_patterns[anchor_type]))
        
        return successful_patterns

class EnhancedMemoryManager:
    """Extended MemoryManager with context utilization capabilities"""
    
    def __init__(self):
        """Initialize enhanced memory manager with ChromaDB"""
        self.client = chromadb.Client()
        self.prompt_memory = PromptMemory()
        
        # Initialize collections
        self.memory_collection = self.client.create_collection(
            name="memory_patterns",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.pattern_collection = self.client.create_collection(
            name="trigger_patterns",
            metadata={"hnsw:space": "cosine"}
        )

    def store_memory_pattern(self, task: str,
                           prompt_responses: List[Dict],
                           effectiveness_ratings: Dict[str, int],
                           memory_triggers: Dict[str, List[str]]) -> None:
        """Store memory patterns with enhanced context and trigger tracking"""
        timestamp = datetime.now().isoformat()
        
        # Group responses by memory anchor type
        anchor_patterns = {
            "location": [],
            "time": [],
            "activity": [],
            "sensory": []
        }
        
        for pr in prompt_responses:
            anchor_type = pr.get('type', 'other')
            if anchor_type in anchor_patterns:
                anchor_patterns[anchor_type].append({
                    'prompt': pr['prompt'],
                    'response': pr['response'],
                    'effectiveness': effectiveness_ratings.get(pr['prompt'], 0),
                    'triggered_memories': memory_triggers.get(pr['prompt'], [])
                })

        # Store main memory entry
        main_metadata = {
            'timestamp': timestamp,
            'task_type': 'memory_recall',
            'anchor_patterns': json.dumps(anchor_patterns),
            'average_effectiveness': sum(effectiveness_ratings.values()) / len(effectiveness_ratings),
            'memory_chain': self._extract_memory_chain(memory_triggers)
        }

        self.memory_collection.add(
            documents=[task],
            metadatas=[main_metadata],
            ids=[f"memory_{timestamp}"]
        )

        # Store pattern information
        self._store_pattern_information(anchor_patterns, timestamp, task)

    def retrieve_memory_patterns(self, task: str, limit: int = 5) -> Dict:
        """Retrieve relevant memory patterns and their effectiveness"""
        similar_results = self.memory_collection.query(
            query_texts=[task],
            n_results=limit
        )

        pattern_results = self.pattern_collection.query(
            query_texts=[task],
            n_results=limit
        )

        successful_patterns = self._analyze_pattern_effectiveness(
            similar_results, pattern_results
        )

        return {
            'similar_sessions': self._process_similar_sessions(similar_results),
            'effective_patterns': successful_patterns,
            'suggested_sequence': self._generate_prompt_sequence(successful_patterns)
        }

    def _store_pattern_information(self, anchor_patterns: Dict, 
                                 timestamp: str, task: str) -> None:
        """Store pattern information for effective memory triggers"""
        for anchor_type, patterns in anchor_patterns.items():
            effective_patterns = [p for p in patterns if p['effectiveness'] >= 4]
            if effective_patterns:
                pattern_metadata = {
                    'anchor_type': anchor_type,
                    'timestamp': timestamp,
                    'original_task': task,
                    'trigger_count': len(effective_patterns),
                    'average_effectiveness': sum(p['effectiveness'] for p in effective_patterns) / len(effective_patterns)
                }
                
                pattern_text = json.dumps([{
                    'prompt_template': p['prompt'],
                    'trigger_type': self._analyze_trigger_type(p['response']),
                    'memory_connections': len(p['triggered_memories'])
                } for p in effective_patterns])

                self.pattern_collection.add(
                    documents=[pattern_text],
                    metadatas=[pattern_metadata],
                    ids=[f"pattern_{anchor_type}_{timestamp}"]
                )

    def _extract_memory_chain(self, memory_triggers: Dict[str, List[str]]) -> List[Dict]:
        """Extract the chain of memory activations"""
        chain = []
        for prompt, triggered_memories in memory_triggers.items():
            chain.append({
                'trigger': prompt,
                'activated_memories': triggered_memories,
                'timestamp': datetime.now().isoformat()
            })
        return chain

    def _analyze_trigger_type(self, response: str) -> str:
        """Analyze the type of memory trigger from the response"""
        trigger_keywords = {
            'visual': ['saw', 'looked', 'screen', 'color', 'ide', 'editor'],
            'spatial': ['where', 'place', 'room', 'desk', 'office'],
            'temporal': ['when', 'time', 'before', 'after', 'during'],
            'procedural': ['did', 'made', 'created', 'built', 'wrote'],
            'emotional': ['felt', 'frustrated', 'excited', 'worried']
        }
        
        response_lower = response.lower()
        trigger_counts = {
            t_type: sum(1 for word in keywords if word in response_lower)
            for t_type, keywords in trigger_keywords.items()
        }
        
        return max(trigger_counts.items(), key=lambda x: x[1])[0]

    def _process_similar_sessions(self, results: Dict) -> List[Dict]:
        """Process and format similar session results"""
        sessions = []
        for idx, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
            if metadata:
                sessions.append({
                    'task': doc,
                    'timestamp': metadata.get('timestamp'),
                    'effectiveness': metadata.get('average_effectiveness', 0),
                    'patterns': json.loads(metadata.get('anchor_patterns', '{}'))
                })
        return sessions

    def _generate_prompt_sequence(self, pattern_effectiveness: Dict) -> List[str]:
        """Generate optimal prompt sequence based on pattern effectiveness"""
        sequence = []
        sorted_types = sorted(
            pattern_effectiveness.items(),
            key=lambda x: x[1]['average_effectiveness'],
            reverse=True
        )
        
        for anchor_type, data in sorted_types:
            patterns = data['patterns']
            if patterns:
                best_patterns = sorted(
                    patterns,
                    key=lambda x: x.get('effectiveness', 0),
                    reverse=True
                )[:2]
                sequence.extend([p['prompt_template'] for p in best_patterns])
        
        return sequence