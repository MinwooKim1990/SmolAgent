# %%
import time
import matplotlib.pyplot as plt
import numpy as np
from game_prev import ImprovedGraphRAG
from game_norm import StandardRAG
from datasets import load_dataset
import re
import string
from collections import Counter
import random

# For BLEU score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# For ROUGE score
from rouge_score import rouge_scorer

# ------------------------------
# Commercial baseline values (realistic values based on current state-of-the-art systems)
COMMERCIAL_BASELINE_PROCESSING_TIME = 2.5  # seconds (Lower is Better)
COMMERCIAL_BASELINE_F1 = {
    'factual': 0.82,        # 단순 사실 관계 질문
    'inferential': 0.75,    # 추론이 필요한 질문
    'complex': 0.70         # 복잡한 관계나 다중 정보가 필요한 질문
}
COMMERCIAL_BASELINE_BLEU = {
    'factual': 0.45,
    'inferential': 0.40,
    'complex': 0.35
}
COMMERCIAL_BASELINE_ROUGE = {
    'factual': 0.50,
    'inferential': 0.45,
    'complex': 0.40
}
# ------------------------------

def load_nq_samples(num_samples):
    """
    RAG 평가를 위한 상세한 샘플을 반환합니다.
    각 컨텍스트마다 3개의 질문과 상세한 답변을 포함합니다.
    질문 유형:
    1. Factual: 텍스트에서 직접적으로 찾을 수 있는 사실 기반 질문
    2. Inferential: 텍스트의 여러 부분을 연결하여 추론이 필요한 질문
    3. Complex: 여러 개념을 종합하거나 텍스트 범위를 넘어선 응용이 필요한 질문
    """
    predefined_samples = [
        # 샘플 1: 인공지능과 기계학습 (한글)
        {
            "context": """인공지능(AI)과 기계학습은 현대 컴퓨터 과학의 핵심 분야입니다. 기계학습은 크게 지도학습, 비지도학습, 강화학습으로 나눌 수 있습니다. 
            지도학습은 레이블이 있는 데이터를 사용하여 모델을 훈련시키는 방법으로, 분류와 회귀 문제를 해결하는 데 주로 사용됩니다. 예를 들어, 이미지 분류, 스팸 메일 감지, 주가 예측 등에 활용됩니다. 
            비지도학습은 레이블이 없는 데이터에서 패턴을 찾는 방법으로, 클러스터링, 차원 축소, 이상 감지 등에 사용됩니다. K-means 클러스터링, 주성분 분석(PCA), 오토인코더 등이 대표적인 예시입니다.
            강화학습은 에이전트가 환경과 상호작용하면서 시행착오를 통해 학습하는 방법입니다. 에이전트는 각 상태에서 행동을 선택하고, 그 결과로 받는 보상을 최대화하는 방향으로 학습합니다. 
            딥러닝은 기계학습의 한 분야로, 인공신경망을 사용하여 데이터의 특징을 자동으로 학습합니다. 특히 컴퓨터 비전, 자연어 처리, 음성 인식 등에서 혁신적인 성과를 보여주고 있습니다.
            최근에는 트랜스포머 구조를 기반으로 한 대규모 언어 모델(LLM)이 주목받고 있으며, GPT, BERT 등이 대표적입니다. 이러한 모델들은 자연어 이해와 생성 능력이 뛰어나 다양한 응용 분야에서 활용되고 있습니다.""",
            "questions": [
                {
                    "question": "기계학습의 세 가지 주요 유형을 나열하시오.", # Factual
                    "answer": "기계학습의 세 가지 주요 유형은 지도학습, 비지도학습, 강화학습입니다."
                },
                {
                    "question": "지도학습과 비지도학습의 차이점을 설명하고, 각각이 어떤 상황에서 더 적합한지 분석하시오.", # Inferential
                    "answer": "지도학습은 레이블이 있는 데이터를 사용하여 분류와 회귀 문제를 해결하는 방법으로, 결과가 명확한 예측 작업(이미지 분류, 스팸 감지 등)에 적합합니다. 반면 비지도학습은 레이블이 없는 데이터에서 패턴을 찾는 방법으로, 데이터의 숨겨진 구조를 발견해야 하는 경우(클러스터링, 이상 감지 등)에 더 적합합니다."
                },
                {
                    "question": "현재 소개된 기계학습 방법들의 한계점은 무엇이며, 이를 극복하기 위한 미래의 발전 방향을 제시하시오.", # Complex
                    "answer": "현재 기계학습 방법들은 각각 한계를 가지고 있습니다. 지도학습은 대량의 레이블된 데이터가 필요하고, 비지도학습은 결과의 해석이 어려우며, 강화학습은 실제 환경에서의 시행착오가 제한적입니다. 이러한 한계를 극복하기 위해서는 적은 데이터로도 학습 가능한 few-shot learning, 해석 가능한 AI, 시뮬레이션 기반 학습 등의 방향으로 발전이 필요합니다. 최근 등장한 트랜스포머 기반 모델들이 이러한 한계 극복의 가능성을 보여주고 있습니다."
                }
            ]
        },
        # 샘플 2: 기후변화와 지구온난화 (한글)
        {
            "context": """기후변화와 지구온난화는 21세기 인류가 직면한 가장 심각한 환경 문제 중 하나입니다. 지구온난화는 대기 중 온실가스 농도 증가로 인해 지구의 평균 기온이 상승하는 현상을 말합니다. 
            주요 온실가스에는 이산화탄소(CO2), 메탄(CH4), 아산화질소(N2O) 등이 있으며, 특히 이산화탄소는 화석연료 사용과 산업화로 인해 급격히 증가하고 있습니다. 
            산업혁명 이후 지구의 평균 기온은 약 1.1°C 상승했으며, 이로 인해 극지방의 빙하가 녹고, 해수면이 상승하며, 이상기후 현상이 빈번하게 발생하고 있습니다.
            기후변화에 대응하기 위해 국제사회는 다양한 노력을 기울이고 있습니다. 2015년 파리협정에서는 지구 평균기온 상승을 산업화 이전 대비 2°C보다 낮은 수준으로 유지하고, 
            나아가 1.5°C로 제한하기 위해 노력하기로 합의했습니다. 이를 위해 각국은 온실가스 감축목표를 설정하고, 재생에너지 확대, 에너지 효율 개선, 탄소중립 정책 등을 추진하고 있습니다.
            기후변화 대응을 위한 기술적 해결책으로는 태양광, 풍력 등 재생에너지 기술, 전기차와 수소차 같은 친환경 운송수단, 탄소포집저장기술(CCS) 등이 있습니다. 
            또한 개인 차원에서도 에너지 절약, 재활용, 친환경 제품 사용 등을 통해 온실가스 감축에 동참할 수 있습니다.""",
            "questions": [
                {
                    "question": "파리협정에서 정한 지구 평균기온 상승 제한 목표는 몇 도입니까?", # Factual
                    "answer": "파리협정에서는 지구 평균기온 상승을 산업화 이전 대비 2°C보다 낮은 수준으로 유지하고, 나아가 1.5°C로 제한하기로 합의했습니다."
                },
                {
                    "question": "기후변화가 극지방 생태계에 미치는 영향과 이로 인한 전 지구적 결과를 설명하시오.", # Inferential
                    "answer": "기후변화로 인해 극지방의 빙하가 녹으면서 직접적으로는 해수면 상승을 초래하고, 이는 연쇄적으로 저지대 침수, 해안 생태계 파괴, 해류 변화 등을 야기합니다. 또한 빙하 감소는 지구의 반사율을 낮춰 온난화를 가속화하는 악순환을 만들어냅니다. 이는 텍스트에 명시된 해수면 상승과 이상기후 현상의 증가로 이어집니다."
                },
                {
                    "question": "현재의 기후변화 대응 정책과 기술들의 효과성을 평가하고, 보다 효과적인 대응 방안을 제시하시오.", # Complex
                    "answer": "현재의 대응책들은 재생에너지, 친환경 운송수단, 탄소포집저장기술 등 기술적 해결책과 국제협약, 개인적 실천 등 다층적 접근을 시도하고 있습니다. 그러나 이러한 노력들은 산업화된 국가들의 경제적 이해관계와 개발도상국의 발전 욕구 사이의 갈등, 기술적 한계, 실행의 지연 등으로 인해 충분한 효과를 거두지 못하고 있습니다. 더 효과적인 대응을 위해서는 국제적 탄소세 도입, 기술 이전 촉진, 순환경제로의 전환 등 보다 강력하고 혁신적인 정책과 함께, AI 기반 기후 모델링, 차세대 에너지 저장 기술 등 첨단 기술의 개발이 필요합니다."
                }
            ]
        },
        # Sample 3: Quantum Computing (영어)
        {
            "context": """Quantum computing represents a revolutionary approach to computation that harnesses the principles of quantum mechanics. Unlike classical computers that use bits (0 or 1), 
            quantum computers use quantum bits or qubits, which can exist in multiple states simultaneously through a phenomenon called superposition. This property, along with quantum entanglement, 
            allows quantum computers to perform certain calculations exponentially faster than classical computers.

            The basic building blocks of quantum computing include quantum gates, which manipulate qubits to perform computations. Common quantum gates include the Hadamard gate (H), 
            which creates superposition, and the CNOT gate, which entangles pairs of qubits. Quantum algorithms leverage these gates to solve specific problems, with Shor's algorithm for 
            factoring large numbers and Grover's algorithm for searching unstructured databases being notable examples.

            However, quantum computers face significant challenges. Quantum decoherence, where qubits lose their quantum properties due to interaction with the environment, 
            requires sophisticated error correction methods. Current quantum computers also need extreme cooling, often to temperatures near absolute zero, to maintain qubit stability. 
            Despite these challenges, companies like IBM, Google, and D-Wave are making significant progress in developing practical quantum computers.

            The potential applications of quantum computing are vast, including cryptography, drug discovery, financial modeling, and climate simulation. In cryptography, 
            quantum computers could potentially break current encryption methods, necessitating the development of quantum-resistant cryptography. In drug discovery, 
            quantum computers could simulate molecular interactions more accurately, potentially accelerating the development of new medicines.""",
            "questions": [
                {
                    "question": "What are the two fundamental quantum properties that quantum computers utilize?", # Factual
                    "answer": "Quantum computers utilize superposition, which allows qubits to exist in multiple states simultaneously, and quantum entanglement."
                },
                {
                    "question": "How might quantum computing's impact on cryptography create both security risks and opportunities? Analyze the potential consequences.", # Inferential
                    "answer": "Quantum computing poses a significant threat to current encryption methods by potentially breaking them, which could compromise existing security systems. However, this challenge also creates opportunities for developing new quantum-resistant cryptography methods. This dual impact requires a proactive approach to security infrastructure, balancing the need to protect current systems while developing new quantum-safe alternatives."
                },
                {
                    "question": "Given the current challenges in quantum computing, evaluate its potential timeline for practical implementation and suggest possible interim solutions for industries awaiting quantum capabilities.", # Complex
                    "answer": "The main challenges of quantum decoherence and the need for extreme cooling suggest that practical, large-scale quantum computers are still years away. Industries should adopt a hybrid approach, combining classical computing with early quantum capabilities where possible. For example, financial institutions could use quantum-inspired algorithms on classical computers while developing quantum-ready systems. Drug discovery could focus on improving classical molecular simulation methods while preparing data structures for future quantum applications. The development timeline should focus on error correction improvements and room-temperature qubit stability before expecting practical implementation."
                }
            ]
        },
        # Sample 4: The Human Brain and Consciousness (영어)
        {
            "context": """The human brain is arguably the most complex structure known to science, containing approximately 86 billion neurons connected through trillions of synapses. 
            These neurons communicate through both electrical and chemical signals, forming intricate networks that give rise to consciousness, thoughts, emotions, and memories. 
            The brain's structure is organized into different regions, each with specialized functions, yet working together in complex ways we are still trying to understand.

            Consciousness, one of the brain's most fascinating properties, remains a subject of intense scientific and philosophical debate. Current theories suggest that consciousness 
            emerges from the integrated processing of information across different brain networks, particularly the default mode network and the frontoparietal network. The global 
            workspace theory proposes that consciousness occurs when information is broadcast widely across the brain, becoming accessible to multiple cognitive systems.

            Memory formation and storage involve multiple brain regions and processes. The hippocampus plays a crucial role in converting short-term memories into long-term memories 
            through a process called consolidation. Long-term potentiation, where neural connections are strengthened through repeated activation, is believed to be the fundamental 
            mechanism behind learning and memory formation. Different types of memories - episodic, semantic, procedural - involve different neural circuits and brain regions.

            Recent advances in neuroscience, including new imaging technologies and research methods, have dramatically increased our understanding of brain function. Techniques like 
            fMRI, EEG, and optogenetics allow researchers to observe and manipulate neural activity in unprecedented ways. These tools have revealed the brain's remarkable plasticity - 
            its ability to reorganize itself by forming new neural connections throughout life, especially in response to learning, experience, or injury.""",
            "questions": [
                {
                    "question": "How many neurons does the human brain contain, and what is their primary method of communication?", # Factual
                    "answer": "The human brain contains approximately 86 billion neurons that communicate through both electrical and chemical signals."
                },
                {
                    "question": "Based on the text's description of memory formation and brain plasticity, explain how learning a new skill might change the brain's structure over time.", # Inferential
                    "answer": "Learning a new skill would involve multiple processes: initial formation of memories through the hippocampus, strengthening of neural connections through long-term potentiation, and the brain's plasticity allowing for the formation of new neural connections. Over time, repeated practice would lead to stronger and more efficient neural circuits specific to that skill, demonstrating the brain's ability to physically reorganize itself in response to learning and experience."
                },
                {
                    "question": "How might understanding the brain's network organization and consciousness theories influence the development of artificial intelligence systems? Propose specific applications.", # Complex
                    "answer": "Understanding the brain's network organization and consciousness theories could revolutionize AI development in several ways. The global workspace theory suggests implementing a central information broadcasting system in AI architectures, similar to how consciousness emerges in the brain. The brain's specialized regions working together could inspire modular AI systems with distinct but interconnected components. Neural plasticity could inform adaptive learning algorithms that reorganize their structure based on experience. This could lead to more human-like AI systems with better generalization abilities and more natural learning processes, though this goes beyond the direct content of the text to apply these principles to AI development."
                }
            ]
        },
        # Sample 5: Evolution and Natural Selection (영어)
        {
            "context": """Evolution by natural selection, first described by Charles Darwin, is the process by which organisms change over time as a result of changes in heritable physical 
            or behavioral traits. These changes allow organisms to better adapt to their environment and help them survive and reproduce more successfully. Natural selection acts on 
            variations within populations, which arise through genetic mutations and genetic recombination during sexual reproduction.

            The process of natural selection requires several key components: variation in traits within a population, heritability of these traits, differential reproduction 
            (some individuals producing more offspring than others), and selection pressure from the environment. Over time, beneficial traits become more common in a population 
            while harmful traits become less common. This process has led to the remarkable diversity of life we see today, from bacteria to blue whales.

            Modern evolutionary theory has been enhanced by our understanding of genetics and molecular biology. DNA sequencing has revealed the genetic basis of inheritance and 
            allowed scientists to track evolutionary changes at the molecular level. We now know that evolution occurs through various mechanisms besides natural selection, 
            including genetic drift, gene flow, and sexual selection. The field of evolutionary developmental biology (evo-devo) has shown how changes in regulatory genes can 
            lead to major evolutionary innovations.

            Evidence for evolution comes from multiple sources, including the fossil record, comparative anatomy, biogeography, and molecular biology. Fossils provide direct 
            evidence of extinct species and show transitional forms between major groups. Comparative anatomy reveals homologous structures that point to common ancestry. 
            DNA and protein sequences show patterns of similarity between species that match their evolutionary relationships. Together, these lines of evidence provide 
            overwhelming support for the theory of evolution.""",
            "questions": [
                {
                    "question": "List the four key components required for natural selection to occur.", # Factual
                    "answer": "The four key components required for natural selection are: variation in traits within a population, heritability of these traits, differential reproduction, and selection pressure from the environment."
                },
                {
                    "question": "How do different types of evidence (fossil record, comparative anatomy, and molecular biology) complement each other in supporting evolutionary theory?", # Inferential
                    "answer": "The different types of evidence complement each other by providing multiple lines of confirmation: fossils provide direct physical evidence of extinct species and transitional forms, comparative anatomy reveals structural similarities suggesting common ancestry, and molecular biology confirms these relationships at the genetic level through DNA and protein sequence similarities. Together, these independent lines of evidence create a robust and consistent picture of evolutionary relationships and mechanisms."
                },
                {
                    "question": "How might our understanding of evolutionary mechanisms be applied to address current global challenges like antibiotic resistance or climate change adaptation? Propose specific strategies.", # Complex
                    "answer": "Our understanding of evolutionary mechanisms could be applied to current challenges in several ways. For antibiotic resistance, understanding how natural selection and genetic mechanisms drive bacterial evolution could help develop new treatment strategies, such as cycling different antibiotics or targeting evolutionary constraints. For climate change adaptation, knowledge of genetic diversity and selection pressures could inform conservation strategies and crop development, focusing on maintaining genetic diversity for adaptation potential and selecting for climate-resilient traits. This would require considering multiple evolutionary mechanisms beyond natural selection, including genetic drift and gene flow, while also accounting for the rapid pace of environmental change compared to natural evolutionary timescales."
                }
            ]
        }
    ]
    
    # 각 컨텍스트의 질문들을 개별 샘플로 변환
    flattened_samples = []
    for sample in predefined_samples:
        for qa in sample["questions"]:
            flattened_samples.append({
                "context": sample["context"],
                "question": qa["question"],
                "answer": qa["answer"]
            })
    
    return flattened_samples[:num_samples]

def plot_processing_time(improved_times, standard_times, baseline):
    indices = np.arange(len(improved_times))
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(indices - width/2, improved_times, width, label="ImprovedGraphRAG")
    plt.bar(indices + width/2, standard_times, width, label="StandardRAG")
    plt.xticks(indices, [f"Sample {i+1}" for i in indices])
    plt.ylabel("Processing Time (seconds)")
    plt.title("Processing Time Benchmark (Lower is Better)")
    plt.axhline(baseline, color="red", linestyle="--", label=f"Commercial Baseline: {baseline:.2f}")
    plt.legend()
    plt.show()

def plot_metric_with_baseline(metric_name, improved_scores, standard_scores, baseline, higher_better=True):
    indices = np.arange(len(improved_scores))
    width = 0.35
    plt.figure(figsize=(10, 6))
    if metric_name == "Exact Match":
        # Use scatter plot for Exact Match due to binary values.
        plt.scatter(indices - width/2, improved_scores, color="blue", label="ImprovedGraphRAG", s=100)
        plt.scatter(indices + width/2, standard_scores, color="orange", label="StandardRAG", s=100)
        plt.plot(indices - width/2, improved_scores, color="blue", linestyle="--")
        plt.plot(indices + width/2, standard_scores, color="orange", linestyle="--")
        plt.xticks(indices, [f"Sample {i+1}" for i in indices])
        plt.ylabel(metric_name)
        plt.ylim(0, 1.1)
    else:
        # Use bar chart for other metrics.
        plt.bar(indices - width/2, improved_scores, width, label="ImprovedGraphRAG")
        plt.bar(indices + width/2, standard_scores, width, label="StandardRAG")
        plt.xticks(indices, [f"Sample {i+1}" for i in indices])
        plt.ylabel(metric_name)
    plt.axhline(baseline, color="red", linestyle="--", label=f"Commercial Baseline: {baseline:.2f}")
    order_text = "Higher is Better" if higher_better else "Lower is Better"
    plt.title(f"{metric_name} Benchmark ({order_text})")
    plt.legend()
    plt.show()

def normalize_answer(s):
    """Lower text, remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return int(pred_tokens == gt_tokens)
    if num_same == 0:
        return 0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)

def compute_metrics_by_type(prediction, ground_truth, question_type):
    """
    질문 유형별로 적절한 메트릭을 계산합니다.
    """
    # F1 Score
    f1 = compute_f1(prediction, ground_truth)
    
    # BLEU Score
    ref_tokens = normalize_answer(ground_truth).split()
    cand_tokens = normalize_answer(prediction).split()
    bleu = sentence_bleu([ref_tokens], cand_tokens, 
                        smoothing_function=SmoothingFunction().method1)
    
    # ROUGE Score
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge = scorer.score(ground_truth, prediction)["rougeL"].fmeasure
    
    return {
        'f1': f1,
        'bleu': bleu,
        'rouge': rouge,
        'type': question_type
    }

def plot_metrics_by_type(improved_metrics, standard_metrics, metric_name):
    """
    질문 유형별로 성능을 시각화합니다.
    """
    question_types = ['factual', 'inferential', 'complex']
    improved_scores = {t: [] for t in question_types}
    standard_scores = {t: [] for t in question_types}
    
    # 유형별로 점수 분류
    for metrics in improved_metrics:
        q_type = metrics['type']
        improved_scores[q_type].append(metrics[metric_name.lower()])
    
    for metrics in standard_metrics:
        q_type = metrics['type']
        standard_scores[q_type].append(metrics[metric_name.lower()])
    
    # 플롯 생성
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'{metric_name} by Question Type')
    
    for i, q_type in enumerate(question_types):
        if len(improved_scores[q_type]) > 0:
            axes[i].boxplot([improved_scores[q_type], standard_scores[q_type]], 
                          tick_labels=['ImprovedGraphRAG', 'StandardRAG'])
            axes[i].set_title(q_type.capitalize())
            axes[i].axhline(y=globals()[f'COMMERCIAL_BASELINE_{metric_name}'][q_type],
                          color='r', linestyle='--', label='Commercial Baseline')
            axes[i].legend()
    
    plt.tight_layout()
    plt.show()


num_tests = 15  # 5개 컨텍스트 * 3개 질문

print(f"\nLoading {num_tests} evaluation samples...")
samples = load_nq_samples(num_tests)

improved_rag = ImprovedGraphRAG()
standard_rag = StandardRAG()

improved_metrics = []
standard_metrics = []
improved_times = []
standard_times = []

# 답변 저장을 위한 리스트
improved_answers = []
standard_answers = []
questions = []
gt_answers = []    # 추가: 정답(ground truth) 저장 리스트

for i, sample in enumerate(samples):
    context = sample["context"]
    question = sample["question"]
    ground_truth = sample["answer"]
    questions.append(question)
    gt_answers.append(ground_truth)    # 추가: gt 답변 저장
    
    # 질문 유형 결정 (인덱스 기반)
    question_type = ['factual', 'inferential', 'complex'][i % 3]
    
    print(f"\n--- Sample {i+1} ({question_type}) ---")
    print("Question:", question)
    
    # ImprovedGraphRAG 처리
    start_time = time.time()
    G, _ = improved_rag.process_story(context)
    answer_improved = improved_rag.answer_question(question, context)
    improved_answers.append(answer_improved)
    duration_improved = time.time() - start_time
    improved_times.append(duration_improved)
    
    # StandardRAG 처리
    start_time = time.time()
    _ = standard_rag.process_story(context)
    answer_standard = standard_rag.answer_question(question, context)
    standard_answers.append(answer_standard)
    duration_standard = time.time() - start_time
    standard_times.append(duration_standard)
    
    # 메트릭 계산
    improved_result = compute_metrics_by_type(answer_improved, ground_truth, question_type)
    standard_result = compute_metrics_by_type(answer_standard, ground_truth, question_type)
    
    improved_metrics.append(improved_result)
    standard_metrics.append(standard_result)
    
    print(f"\nImprovedGraphRAG Metrics:")
    print(f"F1: {improved_result['f1']:.3f}, BLEU: {improved_result['bleu']:.3f}, ROUGE: {improved_result['rouge']:.3f}")
    print(f"StandardRAG Metrics:")
    print(f"F1: {standard_result['f1']:.3f}, BLEU: {standard_result['bleu']:.3f}, ROUGE: {standard_result['rouge']:.3f}")

# 결과 시각화
plot_processing_time(improved_times, standard_times, COMMERCIAL_BASELINE_PROCESSING_TIME)
plot_metrics_by_type(improved_metrics, standard_metrics, 'F1')
plot_metrics_by_type(improved_metrics, standard_metrics, 'BLEU')
plot_metrics_by_type(improved_metrics, standard_metrics, 'ROUGE')

# 종합 결과 출력
print("\n=== Final Results ===")
for q_type in ['factual', 'inferential', 'complex']:
    print(f"\n{q_type.capitalize()} Questions:")
    improved_type = [m for m in improved_metrics if m['type'] == q_type]
    standard_type = [m for m in standard_metrics if m['type'] == q_type]
    
    if improved_type:
        print("ImprovedGraphRAG:")
        print(f"Avg F1: {np.mean([m['f1'] for m in improved_type]):.3f}")
        print(f"Avg BLEU: {np.mean([m['bleu'] for m in improved_type]):.3f}")
        print(f"Avg ROUGE: {np.mean([m['rouge'] for m in improved_type]):.3f}")
    
    if standard_type:
        print("\nStandardRAG:")
        print(f"Avg F1: {np.mean([m['f1'] for m in standard_type]):.3f}")
        print(f"Avg BLEU: {np.mean([m['bleu'] for m in standard_type]):.3f}")
        print(f"Avg ROUGE: {np.mean([m['rouge'] for m in standard_type]):.3f}")

# %%
import json

results = []
for graph, std, question, gt in zip(improved_answers, standard_answers, questions, gt_answers):
    results.append({
        "Question": question,
        "Ground Truth Answer": gt,
        "ImprovedGraphRAG Answer": graph,
        "StandardRAG Answer": std
    })

# JSON 파일로 저장
filename = "benchmark_results.json"
with open(filename, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"benchmark 결과가 {filename} 파일에 저장되었습니다.")

# %%
