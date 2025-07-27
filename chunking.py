from sentence_transformers import SentenceTransformer
import numpy as np

def semantic_chunking(paragraphs, similarity_threshold=0.8):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(paragraphs)
    
    chunks = []
    current_chunk = [paragraphs[0]]
    current_embedding = embeddings[0].reshape(1, -1)
    
    for i in range(1, len(paragraphs)):
        similarity = np.dot(current_embedding, embeddings[i]) / (
            np.linalg.norm(current_embedding) * np.linalg.norm(embeddings[i])
        )
        
        if similarity >= similarity_threshold:
            current_chunk.append(paragraphs[i])
            current_embedding = np.mean([current_embedding, embeddings[i].reshape(1, -1)], axis=0)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [paragraphs[i]]
            current_embedding = embeddings[i].reshape(1, -1)
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks