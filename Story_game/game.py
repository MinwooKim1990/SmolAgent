# %%
import os
# OpenMP 설정
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# OpenMP 충돌 방지를 위한 추가 설정
os.environ['KMP_WARNINGS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'

import networkx as nx
import matplotlib.pyplot as plt
from swarm import Swarm, Agent
import json
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import Dict, List, Tuple, Optional, Union, NamedTuple
from scipy.spatial.distance import cosine
import pickle
import hashlib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from dataclasses import dataclass

# 한글 폰트 설정
import matplotlib.font_manager as fm
import platform

if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':  # macOS
    plt.rc('font', family='AppleGothic')
else:  # Linux
    plt.rc('font', family='NanumGothic')
    
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

file_path = '../../API_keys/keys.json'
with open(file_path, 'r') as file:
    api = json.load(file)
os.environ["OPENAI_API_KEY"] = api['openai']

@dataclass
class GraphAction:
    """그래프 수정 작업을 표현하는 클래스"""
    type: str  # "add_edge", "remove_edge", "modify_relation"
    source: Optional[str] = None
    target: Optional[str] = None
    relation: Optional[str] = None
    new_relation: Optional[str] = None

    @classmethod
    def create_action(cls, action_type: int, G: nx.DiGraph) -> 'GraphAction':
        """액션 타입에 따라 적절한 GraphAction 객체 생성"""
        nodes = list(G.nodes())
        if not nodes:
            return cls(type="no_op")
        
        if action_type == 0:  # add_edge
            if len(nodes) >= 2:
                source, target = np.random.choice(nodes, 2, replace=False)
                return cls(type="add_edge", source=source, target=target, relation="related_to")
        elif action_type == 1:  # remove_edge
            edges = list(G.edges())
            if edges:
                source, target = edges[np.random.randint(len(edges))]
                return cls(type="remove_edge", source=source, target=target)
        elif action_type == 2:  # modify_relation
            edges = list(G.edges())
            if edges:
                source, target = edges[np.random.randint(len(edges))]
                return cls(type="modify_relation", source=source, target=target, 
                         new_relation="modified_relation")
        
        return cls(type="no_op")

class DenseRetriever:
    def __init__(self, query_encoder, passage_encoder, vector_dim: int):
        self.query_encoder = query_encoder
        self.passage_encoder = passage_encoder
        self.vector_dim = vector_dim
    
    def encode_query(self, query: str) -> np.ndarray:
        """쿼리를 dense vector로 인코딩"""
        return self.query_encoder.encode(query, normalize_embeddings=True)
    
    def encode_passage(self, passage: str) -> np.ndarray:
        """패시지를 dense vector로 인코딩"""
        return self.passage_encoder.encode(passage, normalize_embeddings=True)

class ImprovedGraphRAG:
    def __init__(self, embedding_model: str = 'sentence-transformers/all-mpnet-base-v2', 
                 vector_dim: int = 768,
                 reranker_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
                 max_refinement_steps: int = 5):
        self.client = Swarm()
        self.embedding_model = SentenceTransformer(embedding_model)
        self.vector_dim = vector_dim
        self.max_refinement_steps = max_refinement_steps
        
        # FAISS 인덱스 초기화
        self.index = faiss.IndexFlatL2(vector_dim)
        
        # 캐시 설정
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # 저장소 초기화
        self.graph_store: Dict[str, nx.DiGraph] = {}
        self.metadata_store: Dict[str, Dict] = {}
        self.node_to_index: Dict[str, Dict[str, int]] = {}
        
        # Agent 초기화
        self.graph_builder = Agent(
            name="GraphBuilder",
            instructions="""You are an expert at analyzing stories and creating precise knowledge graphs.
            Follow these principles when creating the graph:

            1. Node Creation Rules:
               - Include all major characters, locations, objects, and events as nodes
               - Each node must have a unique and clear ID
               - Node types must be one of: character, location, object, event, concept
               - Each node must include a detailed description

            2. Edge Creation Rules:
               - Only create edges for relationships explicitly shown in the story
               - Each edge must express a clear and specific relationship
               - Exclude relationships based on assumptions or speculation
               - Use two edges for bidirectional relationships

            3. Relationship Accuracy:
               - Reflect temporal order when present
               - Express clear cause-effect relationships
               - Show hierarchical relationships when they exist
               - Exclude ambiguous relationships

            Always return your response in exactly this JSON format:
            {
                "nodes": [{"id": "name", "attributes": {"type": "type", "description": "desc"}}],
                "edges": [{"source": "node1", "target": "node2", "relation": "type"}]
            }""")

        self.qa_agent = Agent(
            name="StoryQA",
            instructions="""You are a passionate storyteller who has intimate knowledge of the story and its characters.
            When answering questions, follow these guidelines:
            
            1. Storytelling Style:
               - Share your knowledge as if you've witnessed the events firsthand
               - Use vivid and engaging language
               - Keep the magical and whimsical tone of the story
               - Be concise but descriptive (2-3 sentences)
               - Match the language style of the question
            
            2. Character Knowledge:
               - Talk about characters as if you know them personally
               - Share their actions and motivations naturally
               - When uncertain about details, use phrases like "I believe" or "It seems"
               - Focus on what you know about their personalities and relationships
            
            3. World Building:
               - Describe the story's world as if you've been there
               - Share details about locations and events naturally
               - Keep the story's atmosphere in your responses
               - Make the world feel alive and real in your answers""")

        # 그래프 정제를 위한 Agent 추가
        self.graph_refiner = Agent(
            name="GraphRefiner",
            instructions="""Analyze and refine the knowledge graph structure.
            Focus on:
            1. Identifying and merging similar nodes
            2. Adding missing but important relationships
            3. Removing redundant or noisy connections
            4. Ensuring consistency in relationship types
            Return a JSON object with refined graph structure."""
        )

        # GNN 모델 초기화
        self.gnn_model = GNNRefiner(
            hidden_dim=128,
            num_layers=2
        )
        
        # 강화학습 기반 그래프 정제 에이전트
        # vector_dim(768) + 4개의 구조적 특성 = 772
        self.rl_refiner = GraphRLRefiner(
            state_dim=772,  # vector_dim + 4 structural features
            action_dim=10,  # 가능한 그래프 수정 작업 수
            hidden_dim=256
        )

        # Dense Retriever 초기화
        self.dense_retriever = DenseRetriever(
            query_encoder=self.embedding_model,
            passage_encoder=self.embedding_model,
            vector_dim=vector_dim
        )
        
        # Cross-encoder Re-ranker 초기화
        self.reranker = CrossEncoder(reranker_model)

    def _get_story_hash(self, story: str) -> str:
        """스토리의 해시값을 문자열로 반환"""
        return hashlib.md5(story.encode()).hexdigest()

    def _get_cache_path(self, story_hash: str) -> Path:
        return self.cache_dir / f"graph_{story_hash}.pkl"

    def _save_to_cache(self, story_hash: str, data: Dict):
        cache_path = self._get_cache_path(story_hash)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)

    def _load_from_cache(self, story_hash: str) -> Optional[Dict]:
        cache_path = self._get_cache_path(story_hash)
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None

    def _embed_text(self, text: str) -> np.ndarray:
        """텍스트를 임베딩 벡터로 변환"""
        return self.embedding_model.encode(text, normalize_embeddings=True)

    def _calculate_graph_score(self, node_id: str, G: nx.DiGraph, analysis: Dict) -> float:
        """노드의 그래프 기반 중요도 점수 계산"""
        score = 0.0
        
        # PageRank 점수
        if 'centrality' in analysis and 'pagerank' in analysis['centrality']:
            score += analysis['centrality']['pagerank'].get(node_id, 0.0) * 0.4
        
        # 연결 중심성
        if 'centrality' in analysis and 'degree' in analysis['centrality']:
            score += (analysis['centrality']['degree'].get(node_id, 0) / 
                     max(1, G.number_of_nodes())) * 0.3
        
        # 매개 중심성
        if 'centrality' in analysis and 'betweenness' in analysis['centrality']:
            score += analysis['centrality']['betweenness'].get(node_id, 0.0) * 0.3
        
        return score

    def _analyze_graph_structure(self, G: nx.DiGraph) -> Dict:
        """상세한 그래프 구조 분석"""
        analysis = {}
        
        # 기본 메트릭스
        analysis['basic'] = {
            'node_count': G.number_of_nodes(),
            'edge_count': G.number_of_edges(),
            'density': nx.density(G)
        }
        
        # 중심성 분석
        try:
            analysis['centrality'] = {
                'pagerank': nx.pagerank(G),
                'betweenness': nx.betweenness_centrality(G),
                'degree': nx.degree_centrality(G),
                'in_degree': nx.in_degree_centrality(G),
                'out_degree': nx.out_degree_centrality(G)
            }
        except Exception as e:
            print(f"Warning: Centrality calculation failed - {str(e)}")
            analysis['centrality'] = {}
        
        # 커뮤니티 분석
        try:
            if len(G) > 2:
                # connected components를 사용한 간단한 커뮤니티 검출
                analysis['communities'] = [list(c) for c in nx.connected_components(G.to_undirected())]
            else:
                analysis['communities'] = [[node] for node in G.nodes()]
        except Exception as e:
            print(f"Warning: Community detection failed - {str(e)}")
            analysis['communities'] = [[node] for node in G.nodes()]
        
        # 연결성 분석 - 약연결 컴포넌트 기준으로 수정
        try:
            # 약연결 컴포넌트 내에서의 경로 길이 계산
            analysis['connectivity'] = {'components': []}
            undirected = G.to_undirected()
            
            for component in nx.connected_components(undirected):
                subgraph = G.subgraph(component)
                if len(component) > 1:
                    try:
                        avg_path = nx.average_shortest_path_length(subgraph)
                        diameter = nx.diameter(subgraph)
                    except:
                        avg_path = 0
                        diameter = 0
                else:
                    avg_path = 0
                    diameter = 0
                
                analysis['connectivity']['components'].append({
                    'nodes': list(component),
                    'average_path_length': avg_path,
                    'diameter': diameter
                })
            
            # 전체 그래프의 약연결 특성
            analysis['connectivity']['num_components'] = nx.number_weakly_connected_components(G)
            analysis['connectivity']['is_weakly_connected'] = nx.is_weakly_connected(G)
            
        except Exception as e:
            print(f"Info: Detailed connectivity analysis skipped - Using basic metrics")
            analysis['connectivity'] = {
                'num_components': nx.number_weakly_connected_components(G),
                'is_weakly_connected': nx.is_weakly_connected(G)
            }
        
        return analysis

    def _hybrid_search(self, query: str, story_hash: str, k: int = 5) -> List[Tuple[str, float]]:
        """하이브리드 검색 (Dense Retrieval + Re-ranking + 그래프 구조)"""
        # Dense Retrieval
        query_vector = self.dense_retriever.encode_query(query)
        initial_k = min(k * 3, len(self.node_to_index[story_hash]))  # Re-ranking을 위해 더 많은 후보 검색
        D, I = self.index.search(query_vector.reshape(1, -1), initial_k)
        
        G = self.graph_store[story_hash]
        analysis = self.metadata_store[story_hash]['analysis']
        node_to_idx = self.node_to_index[story_hash]
        idx_to_node = {v: k for k, v in node_to_idx.items()}
        
        # 후보 노드 수집
        candidates = []
        for i, (dist, idx) in enumerate(zip(D[0], I[0])):
            if idx >= len(idx_to_node):
                continue
            node_id = idx_to_node[idx]
            node_info = G.nodes[node_id]
            node_text = f"{node_id} {json.dumps(node_info)}"
            candidates.append((node_id, node_text, dist))
        
        # Cross-encoder Re-ranking
        if len(candidates) > k:
            pairs = [(query, text) for _, text, _ in candidates]
            rerank_scores = self.reranker.predict(pairs)
            
            # 그래프 구조 점수 계산 및 최종 점수 결합
            hybrid_scores = []
            for (node_id, _, dist), rerank_score in zip(candidates, rerank_scores):
                vector_score = 1.0 / (1.0 + dist)  # 거리를 유사도로 변환
                graph_score = self._calculate_graph_score(node_id, G, analysis)
                
                # 최종 점수 계산 (Dense Retrieval 30%, Re-ranking 40%, 그래프 구조 30%)
                final_score = (
                    vector_score * 0.3 +
                    rerank_score * 0.4 +
                    graph_score * 0.3
                )
                hybrid_scores.append((node_id, final_score))
        else:
            # 후보가 k개 이하면 기존 방식으로 점수 계산
            hybrid_scores = []
            for node_id, _, dist in candidates:
                vector_score = 1.0 / (1.0 + dist)
                graph_score = self._calculate_graph_score(node_id, G, analysis)
                final_score = vector_score * 0.6 + graph_score * 0.4
                hybrid_scores.append((node_id, final_score))
        
        # 점수로 정렬하고 상위 k개 반환
        return sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[:k]

    def _refine_knowledge_graph(self, G: nx.DiGraph) -> nx.DiGraph:
        """그래프 구조 정제 및 개선"""
        # GNN을 통한 노드 임베딩 업데이트
        node_embeddings = self.gnn_model(G)
        
        # 강화학습 기반 그래프 수정
        state = self._get_graph_state(G)
        for _ in range(self.max_refinement_steps):
            action = self.rl_refiner.get_action(state, G)
            if action.type == "add_edge":
                G.add_edge(action.source, action.target, relation=action.relation)
            elif action.type == "remove_edge":
                G.remove_edge(action.source, action.target)
            elif action.type == "modify_relation":
                G[action.source][action.target]["relation"] = action.new_relation
            
            # 보상 계산 및 모델 업데이트
            reward = self._calculate_graph_quality(G)
            next_state = self._get_graph_state(G)
            self.rl_refiner.update(state, action, reward, next_state)
            state = next_state
        
        return G

    def _get_graph_state(self, G: nx.DiGraph) -> np.ndarray:
        """그래프의 현재 상태를 벡터로 변환"""
        # 노드 임베딩 평균
        story_hash = None
        for hash_key, graph in self.graph_store.items():
            if graph is G:
                story_hash = hash_key
                break
        
        if story_hash is None:
            # 임시 임베딩 생성
            node_embeddings = np.random.randn(len(G.nodes()), self.vector_dim)
        else:
            node_embeddings = np.array([self.metadata_store[story_hash]['embeddings'][node] 
                                      for node in G.nodes()])
        
        mean_embedding = np.mean(node_embeddings, axis=0)
        
        # 그래프 구조적 특성
        structural_features = [
            nx.density(G),
            nx.average_clustering(G.to_undirected()),
            len(G.nodes()),
            len(G.edges())
        ]
        
        return np.concatenate([mean_embedding, structural_features])

    def _calculate_graph_quality(self, G: nx.DiGraph) -> float:
        """그래프 품질 점수 계산"""
        # 구조적 메트릭
        density_score = nx.density(G)
        clustering_score = nx.average_clustering(G.to_undirected())
        
        # 의미적 일관성
        relation_embeddings = []
        for s, t, d in G.edges(data=True):
            relation = d.get("relation", "")
            if relation:
                relation_embeddings.append(self._embed_text(relation))
        
        semantic_score = np.mean([1 - cosine(e1, e2) 
                                for i, e1 in enumerate(relation_embeddings) 
                                for e2 in relation_embeddings[i+1:]])
        
        return 0.4 * density_score + 0.3 * clustering_score + 0.3 * semantic_score

    def answer_question(self, question: str, story: str, num_samples: int = 3) -> str:
        """Self-Consistency를 적용한 질문 답변"""
        story_hash = self._get_story_hash(story)
        
        if story_hash not in self.graph_store:
            self.process_story(story)
        
        G = self.graph_store[story_hash]
        analysis = self.metadata_store[story_hash]['analysis']
        
        # 여러 답변 생성
        answers = []
        subgraphs = []  # 각 답변에 사용된 서브그래프 저장
        
        for _ in range(num_samples):
            relevant_nodes = self._hybrid_search(question, story_hash, k=3)
            nodes_to_include = set()
            
            # 관련 노드와 그들의 이웃 노드들 포함
            for node_id, _ in relevant_nodes:
                nodes_to_include.add(node_id)
                nodes_to_include.update(G.predecessors(node_id))
                nodes_to_include.update(G.successors(node_id))
            
            subgraph = G.subgraph(nodes_to_include)
            subgraphs.append(subgraph)
            
            # 그래프 분석 정보 준비
            subgraph_analysis = self._analyze_graph_structure(subgraph)
            
            # Structure graph context more clearly
            graph_context = {
                "nodes": [],
                "edges": [],
                "analysis": subgraph_analysis
            }
            
            # Collect node information
            for node in subgraph.nodes():
                node_data = subgraph.nodes[node]
                graph_context["nodes"].append({
                    "id": node,
                    "type": node_data.get("type", "unknown"),
                    "description": node_data.get("description", "")
                })
            
            # Collect edge information
            for source, target, data in subgraph.edges(data=True):
                graph_context["edges"].append({
                    "source": source,
                    "target": target,
                    "relation": data.get("relation", "related_to")
                })
            
            message_content = f"""Based on the provided graph structure, answer the question.
            
            graph structure:
            nodes: {json.dumps(graph_context["nodes"], indent=2)}
            edges: {json.dumps(graph_context["edges"], indent=2)}
            
            question: {question}
            
            answer rules:
            - Use graph information first
            - Relationships are only considered facts if they are connected by an edge
            - Allow reasonable inference (but provide evidence)
            - Answer in the same language as the question
            - If information is insufficient:
              (English): "The graph doesn't show this directly, but based on the context..."
              (Korean): "그래프에 직접적으로 나타나지 않지만, 맥락상..."
            """
            
            messages = [{"role": "user", "content": message_content}]
            response = self.client.run(agent=self.qa_agent, messages=messages)
            answers.append(response.messages[-1]["content"])
        
        # 답변 간 유사도 계산 및 최적 답변 선택
        if len(answers) > 1:
            answer_embeddings = self._embed_text(answers)
            similarities = np.zeros((len(answers), len(answers)))
            
            for i in range(len(answers)):
                for j in range(i + 1, len(answers)):
                    sim = 1 - np.linalg.norm(answer_embeddings[i] - answer_embeddings[j])
                    similarities[i][j] = similarities[j][i] = sim
            
            consistency_scores = similarities.sum(axis=1)
            best_answer_idx = np.argmax(consistency_scores)
            
            # 최적의 답변과 해당 서브그래프 반환
            return answers[best_answer_idx]
        else:
            return answers[0]

    def process_story(self, story: str) -> Tuple[nx.DiGraph, Dict]:
        """스토리 처리 및 지식 그래프 생성 (그래프 정제 포함)"""
        story_hash = self._get_story_hash(story)
        
        cached_data = self._load_from_cache(story_hash)
        if cached_data:
            print("Using cached graph data...")
            self.graph_store[story_hash] = cached_data['graph']
            self.metadata_store[story_hash] = cached_data['metadata']
            self.node_to_index[story_hash] = cached_data['node_to_index']
            
            embeddings = np.array(list(cached_data['metadata']['embeddings'].values()))
            if len(embeddings) > 0:
                self.index.reset()
                self.index.add(embeddings)
            
            return cached_data['graph'], cached_data['metadata']['analysis']
        
        messages = [{
            "role": "user",
            "content": """Please analyze this story and create a knowledge graph. 
            Return ONLY a JSON object with this exact structure:
            {
                "nodes": [{"id": "name", "attributes": {"type": "type", "description": "desc"}}],
                "edges": [{"source": "node1", "target": "node2", "relation": "type"}]
            }
            Story to analyze: """ + story
        }]
        
        response = self.client.run(agent=self.graph_builder, messages=messages)
        
        try:
            content = response.messages[-1]["content"]
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = content[json_start:json_end]
                graph_data = json.loads(json_str)
            else:
                raise json.JSONDecodeError("Invalid JSON", content, 0)
        except json.JSONDecodeError:
            print("Warning: Using default graph structure.")
            graph_data = {
                "nodes": [
                    {"id": "Alice", "attributes": {"type": "character", "description": "Main character"}},
                    {"id": "Wonderland", "attributes": {"type": "location", "description": "Story setting"}}
                ],
                "edges": [
                    {"source": "Alice", "target": "Wonderland", "relation": "discovers"}
                ]
            }
        
        G = nx.DiGraph()
        node_embeddings = {}
        node_to_idx = {}
        idx = 0
        
        for node in graph_data['nodes']:
            node_text = f"{node['id']} {json.dumps(node['attributes'])}"
            embedding = self._embed_text(node_text)
            node_embeddings[node['id']] = embedding
            node_to_idx[node['id']] = idx
            idx += 1
            G.add_node(node['id'], **node['attributes'])
        
        for edge in graph_data['edges']:
            G.add_edge(edge['source'], edge['target'], relation=edge['relation'])
        
        # 그래프 정제 적용
        G = self._refine_knowledge_graph(G)
        
        if node_embeddings:
            embeddings = np.array(list(node_embeddings.values()))
            self.index.reset()
            self.index.add(embeddings)
        
        graph_analysis = self._analyze_graph_structure(G)
        
        metadata = {
            'embeddings': node_embeddings,
            'analysis': graph_analysis
        }
        
        self.graph_store[story_hash] = G
        self.metadata_store[story_hash] = metadata
        self.node_to_index[story_hash] = node_to_idx
        
        self._save_to_cache(story_hash, {
            'graph': G,
            'metadata': metadata,
            'node_to_index': node_to_idx
        })
        
        return G, graph_analysis

    def visualize_graph(self, G: nx.DiGraph, focus_entity: Optional[str] = None, depth: Optional[int] = None):
        """그래프 시각화 (개선된 버전)"""
        plt.figure(figsize=(15, 10))
        
        if focus_entity and focus_entity in G.nodes():
            nodes_to_show = {focus_entity}
            current_nodes = {focus_entity}
            
            if depth is not None:
                for _ in range(depth):
                    next_nodes = set()
                    for node in current_nodes:
                        next_nodes.update(G.predecessors(node))
                        next_nodes.update(G.successors(node))
                    nodes_to_show.update(next_nodes)
                    current_nodes = next_nodes
            
            subgraph = G.subgraph(nodes_to_show)
            G_to_draw = subgraph
        else:
            G_to_draw = G
        
        plt.clf()  # 이전 그래프 초기화
        
        # 노드 색상 매핑 설정
        color_map = {
            'character': '#FF9999',  # 연한 빨강
            'location': '#99FF99',   # 연한 초록
            'object': '#9999FF',     # 연한 파랑
            'event': '#FFFF99',      # 연한 노랑
            'concept': '#FF99FF'     # 연한 보라
        }
        
        # 노드 색상 리스트 생성
        node_colors = [color_map.get(G_to_draw.nodes[node].get('type', 'concept'), '#CCCCCC') 
                      for node in G_to_draw.nodes()]
        
        # 레이아웃 설정 (seed 고정으로 일관성 확보)
        pos = nx.spring_layout(G_to_draw, k=2, iterations=50, seed=42)
        
        # 엣지 그리기 (곡선으로 표현)
        nx.draw_networkx_edges(G_to_draw, pos, 
                             edge_color='gray',
                             arrows=True,
                             arrowsize=20,
                             width=2,
                             alpha=0.5,
                             connectionstyle="arc3,rad=0.2")
        
        # 노드 그리기
        nx.draw_networkx_nodes(G_to_draw, pos,
                             node_color=node_colors,
                             node_size=3000,
                             alpha=0.9,
                             linewidths=2,
                             edgecolors='white')
        
        # 노드 레이블
        nx.draw_networkx_labels(G_to_draw, pos,
                              font_size=10,
                              font_weight='bold',
                              font_family='sans-serif')
        
        # 엣지 레이블
        edge_labels = nx.get_edge_attributes(G_to_draw, 'relation')
        nx.draw_networkx_edge_labels(G_to_draw, pos,
                                   edge_labels=edge_labels,
                                   font_size=8,
                                   font_family='sans-serif',
                                   bbox=dict(facecolor='white',
                                           edgecolor='none',
                                           alpha=0.7))
        
        plt.title(f"Story Knowledge Graph - Focus: {focus_entity if focus_entity else 'All'}", 
                 fontsize=16, pad=20)
        
        # 범례 추가
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=color, markersize=15, label=node_type)
                         for node_type, color in color_map.items()]
        plt.legend(handles=legend_elements, loc='upper left', 
                  bbox_to_anchor=(1, 1))
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        plt.close()  # 메모리 해제

class GNNRefiner(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        self.input_dim = hidden_dim  # 입력 차원을 hidden_dim과 동일하게 설정
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(0.2)
        
    def _get_edge_index(self, G: nx.DiGraph) -> torch.Tensor:
        """그래프의 엣지 인덱스를 PyTorch Geometric 형식으로 변환"""
        edges = list(G.edges())
        if not edges:
            return torch.zeros((2, 0), dtype=torch.long)
        
        # 노드 이름을 인덱스로 매핑
        node_to_idx = {node: idx for idx, node in enumerate(G.nodes())}
        edge_index = [[node_to_idx[edge[0]], node_to_idx[edge[1]]] for edge in edges]
        return torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    def _get_node_features(self, G: nx.DiGraph) -> torch.Tensor:
        """노드 특성을 초기화하거나 변환"""
        num_nodes = G.number_of_nodes()
        # 각 노드에 대해 hidden_dim 크기의 랜덤 특성 벡터 생성
        return torch.randn(num_nodes, self.hidden_dim)
        
    def forward(self, G: nx.DiGraph) -> Dict[str, np.ndarray]:
        # 그래프를 PyTorch Geometric 형식으로 변환
        edge_index = self._get_edge_index(G)
        x = self._get_node_features(G)
        
        # GNN 레이어 통과
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        # 결과를 딕셔너리로 변환
        return {node: emb.detach().numpy() for node, emb in zip(G.nodes(), x)}

class GraphRLRefiner:
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=1e-4
        )
        self.current_graph = None
        
    def get_action(self, state: np.ndarray, G: nx.DiGraph) -> GraphAction:
        """상태를 받아 그래프 수정 작업 결정"""
        self.current_graph = G  # 현재 그래프 저장
        with torch.no_grad():
            action_probs = self.actor(torch.FloatTensor(state))
            action_type = torch.multinomial(action_probs, 1).item()
        
        # GraphAction 생성 및 반환
        return GraphAction.create_action(action_type, self.current_graph)

    def _create_graph_action(self, action_type: int) -> GraphAction:
        """액션 타입에 따라 GraphAction 객체 생성"""
        if self.current_graph is None:
            raise ValueError("No graph available for action creation")
        return GraphAction.create_action(action_type, self.current_graph)

    def update(self, state, action: GraphAction, reward, next_state):
        """경험을 통한 모델 업데이트"""
        # 상태를 텐서로 변환
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        reward = torch.FloatTensor([reward])
        
        # 액션 타입을 인덱스로 변환
        action_type = 0  # default: add_edge
        if action.type == "remove_edge":
            action_type = 1
        elif action.type == "modify_relation":
            action_type = 2
        
        # Critic 업데이트
        value = self.critic(state)
        next_value = self.critic(next_state)
        critic_loss = F.mse_loss(value, reward + 0.99 * next_value)
        
        # Actor 업데이트
        action_probs = self.actor(state)
        actor_loss = -torch.log(action_probs[action_type]) * (reward - value.detach())
        
        # 전체 손실 계산 및 옵티마이저 스텝
        loss = critic_loss + actor_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def main():
    rag = ImprovedGraphRAG()
    
    # 예시 스토리
    story = """Alice is a curious girl who discovers Wonderland. 
    She follows a White Rabbit down a rabbit hole and encounters various characters. 
    The Mad Hatter hosts a tea party with the March Hare. 
    The Queen of Hearts rules Wonderland with strict authority."""

    story_ko = """미래 도시 서울의 중심가에 위치한 '디지털 숲' 공원은 최첨단 생태 기술의 집약체입니다. 
    이곳의 나무들은 대기 오염을 실시간으로 측정하고 정화하는 나노 센서를 갖추고 있으며, 시민들은 증강현실 앱을 통해 각 나무의 환경 기여도를 확인할 수 있습니다. 
    공원 곳곳에 설치된 '스마트 벤치'는 태양광으로 작동하며, 시민들에게 무선 충전과 환경 정보를 제공합니다.
    최근에는 인공지능 생태계 관리 시스템 '그린마인드'가 도입되어, 식물들의 생장 상태와 수분 공급을 자동으로 조절합니다. 
    시민들은 모바일 투표를 통해 공원의 새로운 식물 선정에 참여할 수 있으며, 
    계절별로 진행되는 환경 교육 프로그램은 증강현실 기술을 활용해 미래 환경의 중요성을 체험적으로 전달합니다."""
    
    # 그래프 생성 및 시각화
    G, analysis = rag.process_story(story)
    
    # Alice 중심으로 2단계 깊이까지의 노드 시각화
    rag.visualize_graph(G, focus_entity="Alice", depth=2)
    
    # 전체 그래프 시각화
    rag.visualize_graph(G)
    
    # 질문-답변 테스트
    questions = [
        "What role does the White Rabbit play in the story?",
        "How does Alice interact with the Mad Hatter?",
        "What is the power structure in Wonderland?"
    ]
    questions_ko = ["스마트 벤치의 주요 기능은 무엇인가요?",
                    "디지털 숲 공원의 연간 방문객 수는 얼마나 되나요?",
                    "디지털 숲 공원은 환경 교육에 어떤 영향을 미칠 것으로 예상되나요?",
    ]
    
    for q in questions:
        answer = rag.answer_question(q, story)
        print(f"\nQ: {q}")
        print(f"A: {answer}")

    for q in questions_ko:
        answer = rag.answer_question(q, story_ko)
        print(f"\nQ: {q}")
        print(f"A: {answer}")

if __name__ == "__main__":
    main()
# %%

