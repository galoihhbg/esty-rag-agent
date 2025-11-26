import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class RagEngine:
    def __init__(self):
        # Khởi tạo OpenAI Client
        self.ai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Khởi tạo ChromaDB (Persistent - Lưu trên ổ cứng)
        db_path = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
        # chromadb.PersistentClient lưu trên đĩa theo path
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        
        # Tạo hoặc lấy Collection
        collection_name = os.getenv("COLLECTION_NAME", "etsy_knowledge_base")
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name
        )

    def get_embedding(self, text):
        """Chuyển text thành vector (theo model embeddings)"""
        text = text.replace("\n", " ")
        response = self.ai_client.embeddings.create(
            input=[text], 
            model="text-embedding-3-small"
        )
        return response.data[0].embedding

    def add_example(self, user_input, correct_output_json, category="general"):
        """Nạp kiến thức mới vào DB"""
        vector = self.get_embedding(user_input)
        
        # ID unique dựa trên nội dung (chuẩn hơn có thể dùng hash + timestamp)
        doc_id = str(abs(hash(user_input)))
        
        self.collection.upsert(
            ids=[doc_id],
            embeddings=[vector],
            documents=[user_input],
            metadatas=[{
                "output": correct_output_json,
                "category": category
            }]
        )
        return True

    def find_similar_examples(self, user_input, n_results=3):
        """Tìm n ví dụ cũ giống input mới nhất"""
        vector = self.get_embedding(user_input)
        
        results = self.collection.query(
            query_embeddings=[vector],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances', 'ids']
        )
        
        examples = []
        # results fields are lists per query
        if results and 'ids' in results and results['ids']:
            count = len(results['ids'][0])
            for i in range(count):
                examples.append({
                    "input": results.get('documents', [[]])[0][i],
                    "output": results.get('metadatas', [[]])[0][i].get('output', ""),
                    "distance": results.get('distances', [[]])[0][i] if 'distances' in results else 0,
                    "id": results.get('ids', [[]])[0][i]
                })
        return examples