import base64
import os
from typing import List, Dict
import numpy as np
from collections import defaultdict
from .faiss_processing import MyFaiss
'''
"global_frame_id":int17
"video_id":string"V001"
"frame_name":string"keyframe_L21_V001_0001.webp"
"frame_index":int0
"split":string"videos-l21-a"
'''

class TRAKE:
    def __init__(self, bin_clip_file: str, json_path: str):
        """
        Initialize TRAKE system with MyFaiss
        
        Args:
            bin_clip_file: Path to the binary file containing the Faiss index
            json_path: Path to the JSON file containing metadata
        """
        self.faiss_searcher = MyFaiss(bin_clip_file, json_path)
        self.keyframes_base_path = "./src/data/Keyframes" 
        
    def retrieve_top_k(self, query: str, k: int = 200) -> List[Dict]:
        """
        Retrieve top K relevant candidates for a single query
        
        Args:
            query: Text query describing an event
            k: Number of top candidates to retrieve
            
        Returns:
            List of candidate information with scores
        """
        '''
"global_frame_id":int17
"video_id":string"V001"
"frame_name":string"keyframe_L21_V001_0001.webp"
"frame_index":int0
"split":string"videos-l21-a"
'''
        scores, image_ids, infos_query, image_paths = self.faiss_searcher.text_search(query, k)
        
        candidates = []
        for i, (score, img_id, info, image_path) in enumerate(zip(scores, image_ids, infos_query, image_paths)):
            if info is not None:  # Filter out None results
                candidates.append({
                    'faiss_idx': int(img_id),
                    'global_frame_id': info['global_frame_id'],
                    'frame_name': info['frame_name'],
                    'video_id': info['video_id'],
                    'split': info['split'], 
                    'score': float(score),
                    'image_path': image_path
                })
        
        return candidates
    
    def group_by_video(self, candidates_list: List[List[Dict]]) -> Dict:
        """
        Group retrieved candidates by video_id
        
        Args:
            candidates_list: List of lists containing candidates for each event
            
        Returns:
            Dictionary grouped by video_id with event candidates
        """
        video_groups = defaultdict(lambda: [[] for _ in range(len(candidates_list))])
        
        for event_idx, candidates in enumerate(candidates_list):
            for candidate in candidates:
                video_id = candidate['video_id']
                video_groups[video_id][event_idx].append(candidate)
        
        return video_groups
    
    def find_valid_sequences(self, video_groups: Dict, n_events: int) -> List[Dict]:
        """
        Find valid sequences where events occur in temporal order
        
        Args:
            video_groups: Dictionary of videos with grouped events
            n_events: Number of events
            
        Returns:
            List of valid sequences with video and frame information
        """
        valid_sequences = []
        
        for video_id, event_candidates in video_groups.items():
            # Check if all events have candidates in this video
            if any(len(candidates) == 0 for candidates in event_candidates):
                continue
            
            self._find_sequences_recursive(
                video_id, event_candidates, 0, [], 0.0, valid_sequences
            )
        
        return valid_sequences
    
    def _find_sequences_recursive(self, video_id: str, event_candidates: List[List[Dict]], 
                                 event_idx: int, current_sequence: List[Dict], 
                                 current_score: float, valid_sequences: List[Dict]):
        """
        Recursively find valid sequences with temporal constraints
        """
        if event_idx == len(event_candidates):
            # Found a valid sequence
            sequence_info = {
                'video_id': video_id,
                'frames': [frame['frame_name'] for frame in current_sequence],
                'global_frame_ids': [frame['global_frame_id'] for frame in current_sequence],
                'splits': list(set(frame['split'] for frame in current_sequence)),
                'total_score': current_score,
                'frame_details': current_sequence.copy()
            }
            valid_sequences.append(sequence_info)
            return
        
        # Sort candidates by global_frame_id for temporal ordering
        sorted_candidates = sorted(
            event_candidates[event_idx], 
            key=lambda x: x['global_frame_id']
        )
        
        for candidate in sorted_candidates:
            # Check temporal constraint: current frame must come after previous frame
            if event_idx > 0 and candidate['global_frame_id'] <= current_sequence[-1]['global_frame_id']:
                continue
            
            # Add candidate and continue search
            current_sequence.append(candidate)
            self._find_sequences_recursive(
                video_id, event_candidates, event_idx + 1, 
                current_sequence, current_score + candidate['score'], 
                valid_sequences
            )
            current_sequence.pop()
    
    def rank_sequences(self, sequences: List[Dict], top_n: int = 20) -> List[Dict]:
        """
        Rank sequences by relevance score
        
        Args:
            sequences: List of valid sequences
            top_n: Number of top sequences to return
            
        Returns:
            Top ranked sequences
        """
        # Sort by total score (higher is better)
        sequences.sort(key=lambda x: x['total_score'], reverse=True)
        return sequences[:top_n]
    
    def _get_image_base64(self, image_path: str) -> str:
        """
        Get base64 encoded image
        
        Args:
            split: Split folder name
            video_id: Video ID
            frame_name: Frame filename
            
        Returns:
            Base64 encoded image string
        """
        try:
            full_image_path = os.path.join(self.keyframes_base_path,image_path)
            with open(full_image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return ""
    
    def format_response(self, sequences: List[Dict]) -> List[Dict]:
        """
        Format sequences into the required API response format
        
        Args:
            sequences: List of ranked sequences
            
        Returns:
            Formatted response matching the API specification
        """
        response = []
        
        for seq_id, sequence in enumerate(sequences):
            frames = []
            
            for frame_id, frame_detail in enumerate(sequence['frame_details']):
                # Extract frame_key from frame_name (remove extension)
                folder_key = frame_detail['frame_name'].split('.')[0].replace('keyframe_', '').split('_')[1]
                video_key = frame_detail['frame_name'].split('.')[0].replace('keyframe_', '').split('_')[0] + '_' + folder_key
                
                # Get base64 encoded image
                image_b64 = self._get_image_base64(
                    frame_detail['image_path']
                )
                
                frames.append({
                    "id": frame_id,
                    "folder_key": folder_key,
                    "video_key": video_key,
                    "frame_key": frame_key,
                    # "timestamp": 0.0,  # Not available in current metadata
                    "image": image_b64
                })
            
            response.append({
                "id": seq_id,
                "frames": frames
            })
        
        return response
    
    def process_temporal_search(self, queries: List[Dict], top_k: int = 100, 
                              top_results: int = 20) -> List[Dict]:
        """
        Main function to process temporal search queries
        
        Args:
            queries: List of query dictionaries with 'query' key
            top_k: Number of candidates to retrieve per event
            top_results: Number of top results to return
            
        Returns:
            Formatted response matching API specification
        """
        # Extract query strings
        events = [q['query'] for q in queries]
        
        print(f"Processing {len(events)} events...")
        
        # Step 1: Retrieve top-k candidates for each event
        candidates_list = []
        for i, event in enumerate(events):
            print(f"Retrieving candidates for event {i+1}: {event[:50]}...")
            candidates = self.retrieve_top_k(event, top_k)
            candidates_list.append(candidates)
            print(f"Found {len(candidates)} candidates for event {i+1}")
        
        # Step 2: Group by video
        print("Grouping candidates by video...")
        video_groups = self.group_by_video(candidates_list)
        print(f"Found candidates in {len(video_groups)} videos")
        
        # Step 3: Find valid sequences
        print("Finding valid temporal sequences...")
        valid_sequences = self.find_valid_sequences(video_groups, len(events))
        print(f"Found {len(valid_sequences)} valid sequences")
        
        if not valid_sequences:
            return []
        
        # Step 4: Rank sequences
        print("Ranking sequences...")
        ranked_sequences = self.rank_sequences(valid_sequences, top_results)
        
        # Step 5: Format response
        print("Formatting response...")
        formatted_response = self.format_response(ranked_sequences)
        
        return formatted_response


def create_trake_instance(bin_clip_file: str, json_path: str) -> TRAKE:
    """
    Factory function to create TRAKE instance
    
    Args:
        bin_clip_file: Path to the binary file containing the Faiss index
        json_path: Path to the JSON file containing metadata
        
    Returns:
        TRAKE instance
    """
    return TRAKE(bin_clip_file, json_path)