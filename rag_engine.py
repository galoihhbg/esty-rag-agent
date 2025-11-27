import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class RagEngine:
    def __init__(self, test_mode=False):
        """
        Khởi tạo RAG Engine.
        
        Args:
            test_mode: Nếu True, sử dụng in-memory DB và không gọi OpenAI API
        """
        self.test_mode = test_mode
        
        if test_mode:
            # Test mode: sử dụng in-memory ChromaDB, không cần OpenAI
            self.ai_client = None
            self.chroma_client = chromadb.Client()
        else:
            # Production mode: sử dụng OpenAI và Persistent ChromaDB
            self.ai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            db_path = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
            self.chroma_client = chromadb.PersistentClient(path=db_path)
        
        # Tạo hoặc lấy Collection
        collection_name = os.getenv("COLLECTION_NAME", "etsy_knowledge_base")
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name
        )

    def get_embedding(self, text):
        """Chuyển text thành vector (theo model embeddings)"""
        if self.test_mode:
            # Test mode: trả về vector giả định
            # text-embedding-3-small returns 1536 dimensions by default
            import hashlib
            EMBEDDING_DIMENSIONS = 1536
            hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
            return [float((hash_val >> (i % 128)) & 1) for i in range(EMBEDDING_DIMENSIONS)]
        
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