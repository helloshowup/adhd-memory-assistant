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
            return float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
        except:
            return 0.0

    def find_similar_tasks(self, current_task: str, task_history: List[Dict]) -> List[Dict]:
        """Find similar tasks from history based on text similarity"""
        similar_tasks = []
        
        for hist_task in task_history:
            similarity = self.calculate_similarity(current_task, hist_task['task'])
            if similarity >= self.similarity_threshold:
                similar_tasks.append({
                    'task': str(hist_task['task']),
                    'similarity': float(similarity),
                    'prompts': hist_task['prompts'],
                    'effectiveness': float(hist_task.get('effectiveness', 0))
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
            for prompt_data in task.get('prompts', []):
                if float(prompt_data.get('effectiveness', 0)) >= self.success_threshold:
                    for anchor_type in successful_patterns.keys():
                        if anchor_type in str(prompt_data.get('type', '')):
                            successful_patterns[anchor_type].append(str(prompt_data['prompt']))
        
        # Remove duplicates while preserving order
        for anchor_type in successful_patterns:
            successful_patterns[anchor_type] = list(dict.fromkeys(successful_patterns[anchor_type]))
        
        return successful_patterns

class EnhancedMemoryManager:
    """Extended MemoryManager with context utilization capabilities"""
    
    def __init__(self):
        """Initialize enhanced memory manager with ChromaDB"""
        try:
            self.client = chromadb.Client()
            self.prompt_memory = PromptMemory()
            
            # Initialize collections with get_or_create pattern
            try:
                self.memory_collection = self.client.get_collection(
                    name="memory_patterns"
                )
            except ValueError:  # Collection doesn't exist
                self.memory_collection = self.client.create_collection(
                    name="memory_patterns",
                    metadata={"hnsw:space": "cosine"}
                )
            
            try:
                self.pattern_collection = self.client.get_collection(
                    name="trigger_patterns"
                )
            except ValueError:  # Collection doesn't exist
                self.pattern_collection = self.client.create_collection(
                    name="trigger_patterns",
                    metadata={"hnsw:space": "cosine"}
                )
        except Exception as e:
            print(f"Error initializing ChromaDB: {str(e)}")
            raise

    def store_memory_pattern(self, task: str,
                           prompt_responses: List[Dict],
                           effectiveness_ratings: Dict[str, int],
                           memory_triggers: Dict[str, List[str]]) -> None:
        """Store memory patterns with enhanced context and trigger tracking"""
        try:
            timestamp = datetime.now().isoformat()
            
            # Ensure all values are of correct type for ChromaDB
            processed_responses = []
            for pr in prompt_responses:
                processed_responses.append({
                    'prompt': str(pr['prompt']),
                    'response': str(pr['response']),
                    'type': str(pr.get('type', 'other')),
                    'effectiveness': int(effectiveness_ratings.get(pr['prompt'], 0))
                })

            # Group responses by memory anchor type
            anchor_patterns = {
                "location": [],
                "time": [],
                "activity": [],
                "sensory": []
            }
            
            for pr in processed_responses:
                anchor_type = pr['type']
                if anchor_type in anchor_patterns:
                    pattern_data = {
                        'prompt': pr['prompt'],
                        'response': pr['response'],
                        'effectiveness': pr['effectiveness'],
                        'triggered_memories': [str(mem) for mem in memory_triggers.get(pr['prompt'], [])]
                    }
                    anchor_patterns[anchor_type].append(pattern_data)

            # Format metadata for ChromaDB
            memory_chain = [
                {
                    'trigger': str(trigger),
                    'activated_memories': [str(mem) for mem in memories],
                    'timestamp': timestamp
                }
                for trigger, memories in memory_triggers.items()
            ]

            main_metadata = {
                'timestamp': timestamp,
                'task_type': 'memory_recall',
                'anchor_patterns': json.dumps(anchor_patterns),
                'average_effectiveness': float(sum(effectiveness_ratings.values()) / max(len(effectiveness_ratings), 1)),
                'memory_chain': json.dumps(memory_chain)
            }

            # Generate unique ID for ChromaDB
            collection_id = f"memory_{int(datetime.now().timestamp())}"

            # Store in ChromaDB
            self.memory_collection.add(
                documents=[str(task)],
                metadatas=[main_metadata],
                ids=[collection_id]
            )

            # Store pattern information
            self._store_pattern_information(anchor_patterns, timestamp, task)
            
        except Exception as e:
            print(f"Error storing memory pattern: {str(e)}")
            raise

    def _store_pattern_information(self, anchor_patterns: Dict, 
                                 timestamp: str, task: str) -> None:
        """Store pattern information for effective memory triggers"""
        try:
            for anchor_type, patterns in anchor_patterns.items():
                effective_patterns = [p for p in patterns if p['effectiveness'] >= 4]
                if effective_patterns:
                    pattern_metadata = {
                        'anchor_type': str(anchor_type),
                        'timestamp': str(timestamp),
                        'original_task': str(task),
                        'trigger_count': int(len(effective_patterns)),
                        'average_effectiveness': float(
                            sum(p['effectiveness'] for p in effective_patterns) / len(effective_patterns)
                        )
                    }
                    
                    pattern_info = [
                        {
                            'prompt_template': str(p['prompt']),
                            'trigger_type': str(self._analyze_trigger_type(p['response'])),
                            'memory_connections': int(len(p.get('triggered_memories', [])))
                        } 
                        for p in effective_patterns
                    ]

                    collection_id = f"pattern_{anchor_type}_{int(datetime.now().timestamp())}"
                    
                    self.pattern_collection.add(
                        documents=[json.dumps(pattern_info)],
                        metadatas=[pattern_metadata],
                        ids=[collection_id]
                    )
        except Exception as e:
            print(f"Error storing pattern information: {str(e)}")
            raise

    def retrieve_memory_patterns(self, task: str, limit: int = 5) -> Dict:
        """Retrieve relevant memory patterns and their effectiveness"""
        try:
            similar_results = self.memory_collection.query(
                query_texts=[str(task)],
                n_results=min(limit, 10)  # Ensure we don't exceed collection size
            )

            pattern_results = self.pattern_collection.query(
                query_texts=[str(task)],
                n_results=min(limit, 10)
            )

            successful_patterns = self._analyze_pattern_effectiveness(
                similar_results, pattern_results
            )

            return {
                'similar_sessions': self._process_similar_sessions(similar_results),
                'effective_patterns': successful_patterns,
                'suggested_sequence': self._generate_prompt_sequence(successful_patterns)
            }
        except Exception as e:
            print(f"Error retrieving memory patterns: {str(e)}")
            return {
                'similar_sessions': [],
                'effective_patterns': {},
                'suggested_sequence': []
            }

    def _analyze_pattern_effectiveness(self, similar_results: Dict, pattern_results: Dict) -> Dict:
        """Analyze effectiveness of memory patterns"""
        try:
            effectiveness_data = {}
            
            # Process similar results
            for idx, metadata in enumerate(similar_results['metadatas'][0]):
                if metadata:
                    patterns = json.loads(metadata.get('anchor_patterns', '{}'))
                    for anchor_type, type_patterns in patterns.items():
                        if anchor_type not in effectiveness_data:
                            effectiveness_data[anchor_type] = {
                                'total_effectiveness': 0.0,
                                'count': 0,
                                'patterns': []
                            }
                        
                        for pattern in type_patterns:
                            if float(pattern['effectiveness']) >= 4:
                                effectiveness_data[anchor_type]['patterns'].append(pattern)
                                effectiveness_data[anchor_type]['total_effectiveness'] += float(pattern['effectiveness'])
                                effectiveness_data[anchor_type]['count'] += 1
            
            # Calculate averages and format output
            pattern_effectiveness = {}
            for anchor_type, data in effectiveness_data.items():
                if data['count'] > 0:
                    pattern_effectiveness[anchor_type] = {
                        'average_effectiveness': float(data['total_effectiveness'] / data['count']),
                        'patterns': data['patterns']
                    }
            
            return pattern_effectiveness
        except Exception as e:
            print(f"Error analyzing pattern effectiveness: {str(e)}")
            return {}

    def _process_similar_sessions(self, results: Dict) -> List[Dict]:
        """Process and format similar session results"""
        try:
            sessions = []
            if results and 'documents' in results and 'metadatas' in results:
                for idx, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                    if metadata:
                        session_data = {
                            'task': str(doc),
                            'timestamp': str(metadata.get('timestamp', '')),
                            'effectiveness': float(metadata.get('average_effectiveness', 0)),
                            'patterns': json.loads(metadata.get('anchor_patterns', '{}'))
                        }
                        sessions.append(session_data)
            return sessions
        except Exception as e:
            print(f"Error processing similar sessions: {str(e)}")
            return []

    def _analyze_trigger_type(self, response: str) -> str:
        """Analyze the type of memory trigger from the response"""
        try:
            trigger_keywords = {
                'visual': ['saw', 'looked', 'screen', 'color', 'ide', 'editor'],
                'spatial': ['where', 'place', 'room', 'desk', 'office'],
                'temporal': ['when', 'time', 'before', 'after', 'during'],
                'procedural': ['did', 'made', 'created', 'built', 'wrote'],
                'emotional': ['felt', 'frustrated', 'excited', 'worried']
            }
            
            response_lower = str(response).lower()
            trigger_counts = {
                t_type: sum(1 for word in keywords if word in response_lower)
                for t_type, keywords in trigger_keywords.items()
            }
            
            return max(trigger_counts.items(), key=lambda x: x[1])[0]
        except Exception as e:
            print(f"Error analyzing trigger type: {str(e)}")
            return 'other'

    def _generate_prompt_sequence(self, pattern_effectiveness: Dict) -> List[str]:
        """Generate optimal prompt sequence based on pattern effectiveness"""
        try:
            sequence = []
            sorted_types = sorted(
                pattern_effectiveness.items(),
                key=lambda x: float(x[1]['average_effectiveness']),
                reverse=True
            )
            
            for anchor_type, data in sorted_types:
                patterns = data['patterns']
                if patterns:
                    best_patterns = sorted(
                        patterns,
                        key=lambda x: float(x.get('effectiveness', 0)),
                        reverse=True
                    )[:2]
                    sequence.extend([str(p['prompt_template']) for p in best_patterns])
            
            return sequence
        except Exception as e:
            print(f"Error generating prompt sequence: {str(e)}")
            return []