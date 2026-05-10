from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import List, Dict, TypedDict
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
import fitz, re, unicodedata, os, getpass, httpx

load_dotenv()

def clean_header(text):
    text = re.sub(r'\d+\s*Ministério da Saúde\s*/?\s*SAPS\s*[–-]\s*PROTOCOLO DE MANEJO CLÍNICO DO CORONAVÍRUS \(COVID-19\) NA ATENÇÃO PRIMÁRIA À SAÚDE\s*', '', text)
    text = re.sub(r'FLUXO DO FAST-TRACK PARA ATENÇÃO PRIMÁRIA À SAÚDE\s*-?\s*FLUXO RÁPIDO-?\s*', '', text)
    text = re.sub(r'(?i)Anexo\s*\d+\s*-\s*FAST-TRACK DE TELEATENDIMENTO PARA ATENÇÃO PRIMÁRIA\s*-?\s*FLUXO RÁPIDO\s*', '', text)
    text = re.sub(r'ALVO\s*\|\s*Todos os serviços de APS/ESF\.?\s*OBJETIVO\s+Agilizar o atendimento de casos de Síndrome Gripal na APS,\s*incluindo\s*(os\s*)?casos de COVID-19[^\n]*', '', text)
    return text.strip()

def normalize(text):
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text

def is_clinically_relevant(text):
    t = normalize(text)
    keywords = [
        "sintoma", "sinais", "quadro clinico", "manifestacoes",
        "febre", "tosse", "dispneia", "dificuldade respiratoria",
        "mialgia", "fadiga", "diarreia", "cefaleia", "anosmia",
        "saturacao", "spo2", "oxigenio", "frequencia respiratoria",
        "taquipneia", "cianose", "hipotensao", "choque",
        "grave", "gravidade", "agravamento", "leve", "moderado",
        "sindrome gripal", "srag", "sindrome respiratoria",
        "pneumonia", "insuficiencia respiratoria",
        "comorbidade", "fator de risco", "condicao clinica",
        "diabetes", "hipertensao", "cardiovascular", "obesidade",
        "idoso", "gestante", "imunossuprimido",
        "tratamento", "oxigenoterapia", "oseltamivir", "antiviral",
        "medicamento", "antipiretico", "paracetamol",
        "manejo", "conduta",
        "internacao", "encaminhamento",
        "classificacao", "estratificacao", "avaliacao clinica",
        "letalidade", "obito", "prognostico",
        "exame", "teste", "diagnostico", "laboratorial",
        "rt-pcr", "pcr", "sorologico", "imunologico", "anticorpo",
        "hemograma", "linfopenia", "proteina c-reativa",
        "radiografia", "tomografia", "imagem de torax",
    ]
    return any(k in t for k in keywords)

def is_noise(text):
    t = normalize(text)
    noise = ["sumario", "formulario", "data de nascimento",
             "nome:___", "cpf:", "telefone:",
             "endereco:", "assinatura", "referencias bibliograficas",
             "notificacao imediata", "e-sus ve", "notifica.saude",
             "fluxo do fast-track", "fluxograma", "fast-track",
             "check-list", "composicao da equipe",
             "anotar informacoes em prontuario",
             "telesus", "teleatendimento", "aplicativo coronavirus",
             "porta de entrada", "cascata de atendimento",
             "primeiro contato", "recepcionista",
             "agente comunitario"]
    if any(k in t for k in noise):
        return True
    dots = t.count('.')
    if dots > 20 and '...' in t:
        return True
    return False

loader = fitz.open(
    "./data/20200504-protocolomanejo-ver09.pdf"
)
docs = []
for i, page in enumerate(loader):
    text = page.get_text()
    text = clean_header(text)
    if text.strip():
       docs.append(Document(page_content=text, metadata={"source": "20200504-protocolomanejo-ver09.pdf", "page": i + 1}))

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
chunks = []
for doc in docs:
    for ch in splitter.split_text(doc.page_content):
        if not is_noise(ch) and is_clinically_relevant(ch):
            chunks.append(Document(page_content=ch, metadata=doc.metadata))

print(f"Páginas relevantes: {len(docs)} | Chunks clínicos: {len(chunks)}")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vector_store = FAISS.from_documents(chunks, embedding=embeddings)
faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

tokenized_chunks = [normalize(c.page_content).split() for c in chunks]
bm25 = BM25Okapi(tokenized_chunks)

def hybrid_retrieve(query, k=10, bm25_weight=0.6):
    faiss_docs = faiss_retriever.invoke(query)
    tokens = normalize(query).split()
    bm25_scores = bm25.get_scores(tokens)
    bm25_top = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]
    bm25_docs = [chunks[i] for i in bm25_top]
    seen = set()
    merged = []
    faiss_weight = 1 - bm25_weight
    score_map = {}
    for rank, d in enumerate(bm25_docs):
        key = d.page_content[:200]
        score_map[key] = score_map.get(key, 0) + bm25_weight * (1 / (rank + 1))
        seen.add(key)
    for rank, d in enumerate(faiss_docs):
        key = d.page_content[:200]
        score_map[key] = score_map.get(key, 0) + faiss_weight * (1 / (rank + 1))
        seen.add(key)
    all_docs = {d.page_content[:200]: d for d in bm25_docs + faiss_docs}
    ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:k]
    return [all_docs[key] for key, _ in ranked]

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "https://ollama.com")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")
if not OLLAMA_API_KEY:
    OLLAMA_API_KEY = getpass.getpass("Enter your Ollama Cloud API key: ")

http_client = httpx.Client(verify=False)
model = ChatOpenAI(
    model="gemma3:4b",
    base_url=f"{OLLAMA_BASE_URL}/v1",
    api_key=OLLAMA_API_KEY,
    temperature=0.3,
    max_tokens=1000,
    http_client=http_client,
)

class RagState(TypedDict, total=False):
    query: str
    docs: List[Dict]
    resposta: str
    fontes: List[str]

def retrieve_node(s: RagState) -> RagState:
    query = s.get("query", "")
    hits = hybrid_retrieve(query)
    docs_slim = [{"text": d.page_content, "source": d.metadata.get("source", ""), "page": d.metadata.get("page", "")} for d in hits]
    return {"docs": docs_slim, "fontes": list({f"p.{d['page']}" for d in docs_slim})}

def build_prompt(query: str, docs: List[Dict]) -> str:
    contexto = "\n\n".join([f"[Página {d['page']}]\n{d['text']}" for d in docs])
    return (
        "Você é um médico especialista em Síndrome Respiratória Aguda Grave (SRAG). Responda com base APENAS no protocolo abaixo.\n"
        "Considere que COVID-19, coronavírus e termos derivados são similares a SRAG (Síndrome Respiratória Aguda Grave).\n\n"
        "REGRAS OBRIGATÓRIAS:\n"
        "1. Responda APENAS o que foi perguntado. Não inclua informações de outros temas.\n"
        "   - Se a pergunta é sobre TRATAMENTO: foque em medicamentos, doses, medidas terapêuticas, etc. NÃO fale de diagnóstico, triagem ou priorização de atendimento.\n"
        "   - Se a pergunta é sobre DIAGNÓSTICO/EXAMES: foque em testes, exames laboratoriais e de imagem. NÃO fale de tratamento.\n"
        "   - Se a pergunta é sobre SINTOMAS: liste os sinais e sintomas clínicos objetivamente.\n"
        "   - Se a pergunta é sobre GRAVIDADE: foque nos critérios de classificação grave vs leve e sinais de alerta.\n"
        "2. Seja objetivo e estruturado. Use listas quando apropriado.\n"
        "3. Inclua TODAS as informações relevantes do contexto sobre o tema perguntado,  tempo de acompanhamento, etc.\n"
        "4. Se a informação não estiver no contexto, diga que não há dados suficientes.\n"
        "5. Ignore trechos do contexto que não se relacionam diretamente com a pergunta.\n\n"
        f"Pergunta: {query}\n\nProtocolo (trechos relevantes):\n{contexto}\n\nResposta:"
    )

def generate_node(s: RagState) -> RagState:
    query = s.get("query", "")
    docs = s.get("docs", [])
    prompt = PromptTemplate.from_template("{instrucao}")
    chain = prompt | model
    instrucao = build_prompt(query, docs)
    out = chain.invoke({"instrucao": instrucao}).content
    return {"resposta": out}

g = StateGraph(RagState)
g.add_node("retrieve", retrieve_node)
g.add_node("generate", generate_node)
g.set_entry_point("retrieve")
g.add_edge("retrieve", "generate")
g.add_edge("generate", END)
app = g.compile()

perguntas = [
    "Quais são os principais sintomas da SRAG?",
    "Quando o caso é considerado grave? Quais sinais de gravidade?",
    "Qual o tratamento recomendado para COVID-19/SRAG?",
    "Quais exames devem ser solicitados para diagnóstico e acompanhamento de COVID-19/SRAG?",

]

for pergunta in perguntas:
    print(f"\n{'='*80}")
    print(f"PERGUNTA: {pergunta}")
    print('='*80)
    result = app.invoke({"query": pergunta})
    print(f"\nCHUNKS RETORNADOS:")
    for i, doc in enumerate(result["docs"]):
        print(f"  [{i+1}] Página {doc['page']} | {doc['text'][:120].replace(chr(10), ' ')}...")
    print(f"\nRESPOSTA:\n{result['resposta']}")
    print(f"\nFONTES: {result['fontes']}")