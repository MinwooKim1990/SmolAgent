import networkx as nx
import matplotlib.pyplot as plt
from swarm import Swarm, Agent
import json
import os
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple, Optional, Union
from scipy.spatial.distance import cosine
import pickle
import hashlib

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

class ImprovedGraphRAG:
    def __init__(self, embedding_model: str = 'sentence-transformers/all-mpnet-base-v2', vector_dim: int = 768):
        self.client = Swarm()
        self.embedding_model = SentenceTransformer(embedding_model)
        self.vector_dim = vector_dim
        
        # FAISS 인덱스 초기화
        self.index = faiss.IndexFlatL2(vector_dim)
        
        # 캐시 설정
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # 저장소 초기화
        self.graph_store: Dict[str, nx.DiGraph] = {}
        self.metadata_store: Dict[str, Dict] = {}
        self.node_to_index: Dict[str, Dict[str, int]] = {}  # story_hash -> {node_id -> index}
        
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
            instructions="""Answer questions using the provided subgraph and graph analysis.
            Consider node importance, relationship patterns, and semantic context."""
        )

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
        
        # 연결성 분석
        try:
            analysis['connectivity'] = {
                'shortest_paths': dict(nx.all_pairs_shortest_path_length(G)),
                'average_path_length': nx.average_shortest_path_length(G),
                'diameter': nx.diameter(G)
            }
        except Exception as e:
            print(f"Warning: Connectivity analysis failed - {str(e)}")
            analysis['connectivity'] = {}
        
        return analysis

    def _hybrid_search(self, query: str, story_hash: str, k: int = 5) -> List[Tuple[str, float]]:
        """하이브리드 검색 (벡터 유사도 + 그래프 구조)"""
        query_vector = self._embed_text(query)
        
        # 벡터 검색
        D, I = self.index.search(query_vector.reshape(1, -1), k * 2)  # 더 많은 후보 검색
        
        G = self.graph_store[story_hash]
        analysis = self.metadata_store[story_hash]['analysis']
        node_to_idx = self.node_to_index[story_hash]
        idx_to_node = {v: k for k, v in node_to_idx.items()}
        
        # 하이브리드 점수 계산
        hybrid_scores = []
        for i, (dist, idx) in enumerate(zip(D[0], I[0])):
            if idx >= len(idx_to_node):
                continue
                
            node_id = idx_to_node[idx]
            vector_score = 1.0 / (1.0 + dist)  # 거리를 유사도로 변환
            graph_score = self._calculate_graph_score(node_id, G, analysis)
            
            # 최종 점수 계산 (벡터 유사도 60%, 그래프 구조 40%)
            final_score = vector_score * 0.6 + graph_score * 0.4
            hybrid_scores.append((node_id, final_score))
        
        # 점수로 정렬하고 상위 k개 반환
        return sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[:k]

    def process_story(self, story: str) -> Tuple[nx.DiGraph, Dict]:
        """스토리 처리 및 지식 그래프 생성 (캐시 활용)"""
        story_hash = self._get_story_hash(story)
        
        # 캐시 확인
        cached_data = self._load_from_cache(story_hash)
        if cached_data:
            print("Using cached graph data...")
            self.graph_store[story_hash] = cached_data['graph']
            self.metadata_store[story_hash] = cached_data['metadata']
            self.node_to_index[story_hash] = cached_data['node_to_index']
            
            # FAISS 인덱스 복원
            embeddings = np.array(list(cached_data['metadata']['embeddings'].values()))
            if len(embeddings) > 0:
                self.index.reset()
                self.index.add(embeddings)
            
            return cached_data['graph'], cached_data['metadata']['analysis']
        
        # 새로운 그래프 생성 (이전 코드와 동일)
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
        
        # 그래프 생성
        G = nx.DiGraph()
        node_embeddings = {}
        node_to_idx = {}
        idx = 0
        
        # 노드 추가 및 임베딩
        for node in graph_data['nodes']:
            node_text = f"{node['id']} {json.dumps(node['attributes'])}"
            embedding = self._embed_text(node_text)
            node_embeddings[node['id']] = embedding
            node_to_idx[node['id']] = idx
            idx += 1
            G.add_node(node['id'], **node['attributes'])
        
        # FAISS 인덱스 업데이트
        if node_embeddings:
            embeddings = np.array(list(node_embeddings.values()))
            self.index.reset()
            self.index.add(embeddings)
        
        # 엣지 추가
        for edge in graph_data['edges']:
            G.add_edge(edge['source'], edge['target'], relation=edge['relation'])
        
        # 그래프 분석
        graph_analysis = self._analyze_graph_structure(G)
        
        # 메타데이터 저장
        metadata = {
            'embeddings': node_embeddings,
            'analysis': graph_analysis
        }
        
        # 저장소에 저장
        self.graph_store[story_hash] = G
        self.metadata_store[story_hash] = metadata
        self.node_to_index[story_hash] = node_to_idx
        
        # 캐시에 저장
        self._save_to_cache(story_hash, {
            'graph': G,
            'metadata': metadata,
            'node_to_index': node_to_idx
        })
        
        return G, graph_analysis

    def answer_question(self, question: str, story: str) -> str:
        """질문 답변 (하이브리드 검색 활용)"""
        story_hash = self._get_story_hash(story)
        
        # 스토리가 처리되지 않은 경우 처리
        if story_hash not in self.graph_store:
            self.process_story(story)
        
        G = self.graph_store[story_hash]
        analysis = self.metadata_store[story_hash]['analysis']
        
        # 하이브리드 검색으로 관련 노드 찾기
        relevant_nodes = self._hybrid_search(question, story_hash, k=3)
        
        # 서브그래프 추출
        nodes_to_include = set()
        for node_id, _ in relevant_nodes:
            nodes_to_include.add(node_id)
            # 1-hop 이웃 노드 추가
            nodes_to_include.update(G.predecessors(node_id))
            nodes_to_include.update(G.successors(node_id))
        
        subgraph = G.subgraph(nodes_to_include)
        
        # 컨텍스트 준비
        context = {
            "relevant_nodes": [node for node, score in relevant_nodes],
            "subgraph_nodes": dict(subgraph.nodes(data=True)),
            "subgraph_edges": [{"source": s, "target": t, "attributes": d} for s, t, d in subgraph.edges(data=True)],
            "graph_analysis": analysis
        }
        
        # QA Agent에 전달할 메시지 구성
        message_content = f"""Based on the following context, please answer the question.

Context:
{json.dumps(context, indent=2)}

Question: {question}

Please provide a clear and concise answer based on the information available in the context."""

        messages = [{"role": "user", "content": message_content}]
        
        # 답변 생성
        response = self.client.run(agent=self.qa_agent, messages=messages)
        return response.messages[-1]["content"]

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

def main():
    rag = ImprovedGraphRAG()
    
    # 예시 스토리
    story = """Alice is a curious girl who discovers Wonderland. 
    She follows a White Rabbit down a rabbit hole and encounters various characters. 
    The Mad Hatter hosts a tea party with the March Hare. 
    The Queen of Hearts rules Wonderland with strict authority."""
    
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
    
    for q in questions:
        answer = rag.answer_question(q, story)
        print(f"\nQ: {q}")
        print(f"A: {answer}")