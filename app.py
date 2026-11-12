import os
import re
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from groq import Groq
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()
groq_api_key =os.getenv("GROQ_API_KEY")
llm_client = Groq(api_key=groq_api_key)

folder_path ="data/raw"
documents =[]

for file_name in os.listdir(folder_path):
    if file_name.endswith('.pdf'):
        pdf_path =os.path.join(folder_path, file_name)
        reader=PdfReader(pdf_path)
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            clean_text = re.sub(r"\s+"," ",text).strip()
            documents.append({"file_name": file_name,
                              "page_number" :page_number, 
                              "text": clean_text})

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks= []

for document in documents:
    split_text = text_splitter.split_text(document["text"])
    for chunk_text in split_text:
        chunks.append({
            "file_name" : document["file_name"],
            "page_number" : document["page_number"],
            "text" : chunk_text
        })

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

text_for_embedding = []

#DENSE SEARCH - EMBEEDINGS
for chunk_record in chunks:
    text_for_embedding.append(chunk_record["text"])

#BM25-WORDS,TOKENS
tokenized_chunks = []

for text in text_for_embedding:
    tokenized_chunks.append(text.lower().split())

bm25 = BM25Okapi(tokenized_chunks)

embeddings = embedding_model.encode(text_for_embedding)

chroma_client = chromadb.PersistentClient(path="vector_store")

collection = chroma_client.get_or_create_collection(name="aetna_document")

ids =[]

for i in range(len(embeddings)):
    ids.append(f"chunk_{i}")

metadatas =[]
for chunk_record in chunks:
    metadatas.append({
        "file_name": chunk_record["file_name"],
        "page_number": chunk_record["page_number"]
    })

collection.upsert(
    ids=ids,
    embeddings=embeddings.tolist(),
    documents=text_for_embedding,
    metadatas=metadatas
)

'''print(collection.count())'''

#query = input("Ask a question: ")
def answer_question(query):

    tokenized_query = query.lower().split()

    bm25_scores = bm25.get_scores(tokenized_query)

    best_bm25_index= bm25_scores.argmax() #argmax tells position where number in highest
    
    best_bm25_chunk = chunks[best_bm25_index]
    #print(best_bm25_chunk)
    bm25_scores_copy =bm25_scores.copy()
    bm25_ids=[]

    for rank in range(3):
        best_index=bm25_scores_copy.argmax()
        bm25_ids.append(f"chunk_{best_index}")
        bm25_scores_copy[best_index]= -1
    #print(bm25_ids)

    query_embedding = embedding_model.encode(query)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=3
    )

    dense_ids =results["ids"][0]
    #print(dense_ids)

    rrf_scores ={}
    k=60 #k is standard constant for formula 1/(k+rank)

    for rank , chunk_id in enumerate(dense_ids, start=1):
        rrf_scores[chunk_id] = 1 / (k + rank) 


    for rank , chunk_id in enumerate(bm25_ids, start=1):
        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = 0
        rrf_scores[chunk_id] += 1 / (k + rank)
    #print(rrf_scores)

    rrf_scores_copy = rrf_scores.copy()
    hybrid_ids = []

    for rank in range(3):
        best_chunk_id = max(rrf_scores_copy, key=rrf_scores_copy.get)
        hybrid_ids.append(best_chunk_id)
        del rrf_scores_copy[best_chunk_id]

    #print(hybrid_ids)

    rerank_pairs = []

    for chunk_id in hybrid_ids:
        chunk_index=int(chunk_id.split("_")[1])
                        
        chunk_text = chunks[chunk_index]["text"]
        rerank_pairs.append([query, chunk_text])

    rerank_scores = reranker.predict(rerank_pairs)
    #print(rerank_scores)

    rerank_scores_copy = rerank_scores.copy()
    reranked_ids = []

    for rank in range(3):
        best_position = rerank_scores_copy.argmax()
        best_chunk_id = hybrid_ids[best_position]
        reranked_ids.append(best_chunk_id)
        rerank_scores_copy[best_position] = -999
    #print(reranked_ids)

    #fallback hitl
    best_rerank_score = max(rerank_scores)

    needs_human_review = False

    if best_rerank_score < 0:
        needs_human_review = True


    '''retrieved_documents = results["documents"][0]
    retrieved_metadata= results["metadatas"][0]
    context_parts =[]
    for i in range(len(retrieved_documents)):
        source = retrieved_metadata[i]
        context_parts.append(
            f"Source: {source['file_name']}, page {source['page_number']}\nText: {retrieved_documents[i]}"
        )
    context ="\n\n".join(context_parts)'''


    retrieved_documents = []
    retrieved_metadata = []

    for chunk_id in reranked_ids:
        chunk_index = int(chunk_id.split("_")[1])
        chunk_record = chunks[chunk_index]

        retrieved_documents.append(chunk_record["text"])

        retrieved_metadata.append({
            "file_name": chunk_record["file_name"],
            "page_number": chunk_record["page_number"]
        })


    context_parts = []

    for i in range(len(retrieved_documents)):
        source = retrieved_metadata[i]

        context_parts.append(
            f"Source: {source['file_name']}, page {source['page_number']}\n"
            f"Text: {retrieved_documents[i]}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
    Use only the context below to answer the question Include the source file name and page number in the answer. If you don't know the answer, just say you don't know.
    Do not use outside knowledge or make up information.
    If the answer is not supported by the context, say: "I don't have enough information in the provided documents."

    Context : {context}

    Question : {query}
    """

    response =llm_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role" : "user" , "content" : prompt}]
    )

    answer = response.choices[0].message.content
    if needs_human_review:
        answer = answer + "\n\nHuman review recommended."
    #print(answer)
    return answer