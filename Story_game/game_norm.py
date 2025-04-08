# %%
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple, Optional
import faiss
import json
import pickle
import hashlib
from pathlib import Path
from swarm import Swarm, Agent
import os
file_path = '../../API_keys/keys.json'
with open(file_path, 'r') as file:
    api = json.load(file)
os.environ["OPENAI_API_KEY"] = api['openai']

class StandardRAG:
    def __init__(self, embedding_model: str = 'sentence-transformers/all-mpnet-base-v2', vector_dim: int = 768):
        self.client = Swarm()
        self.embedding_model = SentenceTransformer(embedding_model)
        self.vector_dim = vector_dim
        
        # FAISS 인덱스 초기화
        self.index = faiss.IndexFlatL2(vector_dim)
        
        # 캐시 설정
        self.cache_dir = Path("cache_norm")
        self.cache_dir.mkdir(exist_ok=True)
        
        # 저장소 초기화
        self.story_store: Dict[str, str] = {}
        self.metadata_store: Dict[str, Dict] = {}
        self.text_chunks: Dict[str, List[str]] = {}
        
        # Agent 초기화
        self.chunk_agent = Agent(
            name="TextChunker",
            instructions="""You are an expert at analyzing stories and breaking them into meaningful chunks.
            Return a list of text chunks that preserve the story's context and meaning."""
        )
        
        self.qa_agent = Agent(
            name="StoryQA",
            instructions="""Answer questions using the provided context chunks.
            Consider the relevance and importance of each chunk to provide accurate answers."""
        )

    def _get_story_hash(self, story: str) -> str:
        """스토리의 해시값을 문자열로 반환"""
        return hashlib.md5(story.encode()).hexdigest()

    def _get_cache_path(self, story_hash: str) -> Path:
        return self.cache_dir / f"rag_{story_hash}.pkl"

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
        if isinstance(text, list):
            # 이미 리스트인 경우
            return self.embedding_model.encode(text, normalize_embeddings=True)
        # 단일 문자열인 경우
        return self.embedding_model.encode([text], normalize_embeddings=True)[0]

    def _chunk_story(self, story: str) -> List[str]:
        """스토리를 의미 있는 청크로 분할"""
        messages = [{
            "role": "user",
            "content": f"""Please break this story into meaningful chunks. 
            Return ONLY a JSON array of text chunks, nothing else.
            Story to analyze: {story}"""
        }]
        
        response = self.client.run(agent=self.chunk_agent, messages=messages)
        
        try:
            content = response.messages[-1]["content"]
            # JSON 문자열 찾기
            json_start = content.find('[')
            json_end = content.rfind(']') + 1
            if json_start != -1 and json_end != -1:
                json_str = content[json_start:json_end]
                chunks = json.loads(json_str)
                if isinstance(chunks, list) and all(isinstance(chunk, str) for chunk in chunks):
                    return chunks
            # JSON 파싱 실패시 문장 단위로 분할
            return [sent.strip() for sent in story.split('.') if sent.strip()]
        except json.JSONDecodeError:
            # 실패시 문장 단위로 분할
            return [sent.strip() for sent in story.split('.') if sent.strip()]

    def process_story(self, story: str) -> Dict:
        """스토리 처리 및 청크 임베딩"""
        story_hash = self._get_story_hash(story)
        
        # 캐시 확인
        cached_data = self._load_from_cache(story_hash)
        if cached_data:
            print("Using cached data...")
            self.story_store[story_hash] = cached_data['story']
            self.metadata_store[story_hash] = cached_data['metadata']
            self.text_chunks[story_hash] = cached_data['chunks']
            
            # FAISS 인덱스 복원
            embeddings = np.array(cached_data['metadata']['embeddings'])
            if len(embeddings) > 0:
                self.index.reset()
                self.index.add(embeddings)
            
            return cached_data['metadata']
        
        # 새로운 처리
        chunks = self._chunk_story(story)
        chunk_embeddings = []
        
        for chunk in chunks:
            embedding = self._embed_text(chunk)
            chunk_embeddings.append(embedding)
        
        # FAISS 인덱스 업데이트
        embeddings = np.array(chunk_embeddings)
        self.index.reset()
        self.index.add(embeddings)
        
        # 메타데이터 저장
        metadata = {
            'embeddings': embeddings,
            'chunk_count': len(chunks)
        }
        
        # 저장소에 저장
        self.story_store[story_hash] = story
        self.metadata_store[story_hash] = metadata
        self.text_chunks[story_hash] = chunks
        
        # 캐시에 저장
        self._save_to_cache(story_hash, {
            'story': story,
            'metadata': metadata,
            'chunks': chunks
        })
        
        return metadata

    def answer_question(self, question: str, story: str, num_samples: int = 3) -> str:
        """Self-Consistency를 적용한 질문 답변"""
        story_hash = self._get_story_hash(story)
        
        # 스토리가 처리되지 않은 경우 처리
        if story_hash not in self.story_store:
            self.process_story(story)
        
        # 여러 답변 생성
        answers = []
        for _ in range(num_samples):
            # 질문 임베딩
            query_vector = self._embed_text(question)
            
            # 관련 청크 검색
            k = min(3, len(self.text_chunks[story_hash]))
            D, I = self.index.search(query_vector.reshape(1, -1), k)
            
            # 관련 청크 수집
            relevant_chunks = [self.text_chunks[story_hash][i] for i in I[0]]
            
            # 컨텍스트 준비
            context = {
                "relevant_chunks": relevant_chunks,
                "distances": D[0].tolist()
            }
            
            # QA Agent에 전달할 메시지 구성
            message_content = f"""Based on the following context, please answer the question.
            Context: {json.dumps(context, indent=2)}
            Question: {question}"""
            
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
            return answers[best_answer_idx]
        else:
            return answers[0]

def main():
    rag = StandardRAG()
    
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
    
    # 스토리 처리
    metadata = rag.process_story(story)
    print(f"Processed story into {metadata['chunk_count']} chunks")
    
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
