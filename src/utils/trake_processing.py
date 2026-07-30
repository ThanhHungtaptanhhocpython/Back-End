import base64
import os
from typing import List, Dict
import numpy as np
from collections import defaultdict
import math
import os
from .faiss_processing import MyFaiss
from src.services.reranker_service import reranker_service
from src.utils.nlp_processing import QueryPlanner
'''
"global_frame_id":int17
"video_id":string"V001"
"frame_name":string"keyframe_L21_V001_0001.webp"
"frame_index":int0
"split":string"videos-l21-a"
'''

class TRAKE:
    def __init__(self, faiss_searcher: MyFaiss):
        """
        Initialize TRAKE system with MyFaiss
        
        Args:
            faiss_searcher: Existing instance of MyFaiss
        """
        self.faiss_searcher = faiss_searcher
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
                    'timestamp': info.get('timestamp', 0.0),
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
    
    def beam_search_sequences(self, video_id: str, event_candidates: List[List[Dict]], beam_width: int = 50) -> List[Dict]:
        """
        Find top temporal sequences using beam search.
        
        Args:
            video_id: The ID of the video being processed.
            event_candidates: List of candidate lists for each event.
            beam_width: Maximum number of partial sequences to keep at each step.
        """
        # Initialize beam with candidates from the first event
        beam = []
        for candidate in event_candidates[0]:
            seq_info = {
                'video_id': video_id,
                'frames': [candidate['frame_name']],
                'global_frame_ids': [candidate['global_frame_id']],
                'timestamps': [candidate.get('timestamp', 0.0)],
                'splits': [candidate['split']],
                'base_score': candidate['score'],
                'total_score': candidate['score'],
                'frame_details': [candidate]
            }
            beam.append(seq_info)
        
        # Sort beam and keep top B
        beam.sort(key=lambda x: x['total_score'], reverse=True)
        beam = beam[:beam_width]
        
        # Process subsequent events
        for event_idx in range(1, len(event_candidates)):
            new_beam = []
            next_candidates = event_candidates[event_idx]
            
            # Sort next candidates by global_frame_id for temporal checks
            next_candidates.sort(key=lambda x: x['global_frame_id'])
            
            for seq in beam:
                last_frame_id = seq['global_frame_ids'][-1]
                for candidate in next_candidates:
                    # Temporal constraint: current frame must come after previous frame
                    if candidate['global_frame_id'] > last_frame_id:
                        new_base_score = seq['base_score'] + candidate['score']
                        time_gap = candidate.get('timestamp', 0.0) - seq['timestamps'][0]
                        
                        # Exponential decay penalty
                        alpha = 0.01
                        penalty = math.exp(-alpha * time_gap) if time_gap > 0 else 1.0
                        new_total_score = new_base_score * penalty
                        
                        new_seq = {
                            'video_id': video_id,
                            'frames': seq['frames'] + [candidate['frame_name']],
                            'global_frame_ids': seq['global_frame_ids'] + [candidate['global_frame_id']],
                            'timestamps': seq['timestamps'] + [candidate.get('timestamp', 0.0)],
                            'splits': list(set(seq['splits'] + [candidate['split']])),
                            'base_score': new_base_score,
                            'total_score': new_total_score,
                            'frame_details': seq['frame_details'] + [candidate]
                        }
                        new_beam.append(new_seq)
            
            # Keep top B
            new_beam.sort(key=lambda x: x['total_score'], reverse=True)
            beam = new_beam[:beam_width]
            
            if not beam:
                # Beam is empty, no valid temporal paths
                break
                
        return beam

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
            
            sequences = self.beam_search_sequences(video_id, event_candidates, beam_width=50)
            valid_sequences.extend(sequences)
        
        return valid_sequences
    
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
                    "frame_key": frame_detail['global_frame_id'],
                    "timestamp": frame_detail.get('timestamp', 0.0),
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
        
        # Step 4: Rank sequences initially
        print("Ranking sequences...")
        ranked_sequences = self.rank_sequences(valid_sequences, top_results)
        
        # Step 4.5: BLIP-VQA Sequence Validation (Phase 6.4)
        print("Validating Top sequences with BLIP-VQA...")
        for seq in ranked_sequences:
            vqa_scores = []
            for i, frame_detail in enumerate(seq['frame_details']):
                event_query = events[i]
                vqa_question = QueryPlanner.generate_vqa_question(event_query)
                
                frame_name = frame_detail['frame_name']
                split = frame_detail.get('split', '')
                
                # Try to locate the image
                possible_path1 = os.path.join(self.keyframes_base_path, split, frame_name)
                possible_path2 = os.path.join(self.keyframes_base_path, frame_name)
                
                img_path = possible_path1 if os.path.exists(possible_path1) else possible_path2
                if not os.path.exists(img_path):
                    # fallback to frame_detail['image_path'] if it is an absolute path
                    img_path = frame_detail.get('image_path', '')
                    
                score = 0.0
                if os.path.exists(img_path):
                    score = reranker_service.score_image(img_path, vqa_question)
                vqa_scores.append(score)
            
            # Average VQA score across all frames in the sequence
            avg_vqa = sum(vqa_scores) / len(vqa_scores) if vqa_scores else 0.0
            
            # Blend 70% original decayed score + 30% VQA confidence
            # Note: total_score could be > 1.0 (since it's a sum of Faiss scores). 
            # We scale VQA by the number of events to match the magnitude.
            vqa_scaled = avg_vqa * len(events)
            
            old_score = seq['total_score']
            new_score = (old_score * 0.7) + (vqa_scaled * 0.3)
            seq['total_score'] = new_score
            seq['vqa_confidence'] = avg_vqa
        
        # Re-sort after VQA validation
        ranked_sequences.sort(key=lambda x: x['total_score'], reverse=True)
        
        # Step 5: Format response
        print("Formatting response...")
        formatted_response = self.format_response(ranked_sequences)
        
        return formatted_response