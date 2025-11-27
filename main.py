from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, validator
from typing import List, Union, Optional, Any
from rag_engine import RagEngine
from prompt_manager import assemble_prompt
from database import DatabaseManager
from openai import OpenAI
import os
import json

# Khởi tạo App
app = FastAPI(title="Etsy RAG Agent")

# Lazy/conditional initialization to make tests easier
if os.getenv("TESTING", "0") != "1":
    rag = RagEngine()
    ai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    db = DatabaseManager()
else:
    rag = RagEngine(test_mode=True)
    ai_client = None
    db = DatabaseManager(test_mode=True)

# Mount static files
static_path = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

# --- Data Models ---
class ConfigField(BaseModel):
    name: str
    type: str
    options: Optional[dict] = None

class TrainingRequest(BaseModel):
    user_input: str
    correct_output: str # JSON string
    category: str = "general"

class PredictRequest(BaseModel):
    user_input: str
    config_json: Union[str, List[dict]]
    color_list: Union[str, List[str]]

    # Parse config_json into list of ConfigField-like dicts
    @validator("config_json", pre=True)
    def parse_config_json(cls, v):
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
            except json.JSONDecodeError:
                raise ValueError("config_json must be valid JSON string or a list of objects")
            if not isinstance(parsed, list):
                raise ValueError("config_json must be a JSON array of field definitions")
            # Basic shape check
            for idx, item in enumerate(parsed):
                if not isinstance(item, dict) or "name" not in item or "type" not in item:
                    raise ValueError(f"config_json[{idx}] must be an object with at least 'name' and 'type'")
            return parsed
        elif isinstance(v, list):
            for idx, item in enumerate(v):
                if not isinstance(item, dict) or "name" not in item or "type" not in item:
                    raise ValueError(f"config_json[{idx}] must be an object with at least 'name' and 'type'")
            return v
        else:
            raise ValueError("config_json must be a JSON string or a list of objects")

    @validator("color_list", pre=True)
    def parse_color_list(cls, v):
        if isinstance(v, str):
            # split by comma and strip
            return [c.strip() for c in v.split(",") if c.strip()]
        elif isinstance(v, list):
            return [str(c).strip() for c in v if str(c).strip()]
        else:
            raise ValueError("color_list must be a CSV string or a list of strings")

# --- Endpoints ---

@app.post("/train")
async def train_knowledge(data: TrainingRequest):
    """API để dạy AI học các case khó"""
    try:
        # Validate JSON output trước khi lưu
        json.loads(data.correct_output) 
        
        success = rag.add_example(data.user_input, data.correct_output, data.category)
        return {"status": "success", "message": "Example embedded and saved."}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="correct_output must be a valid JSON string")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict_order(data: PredictRequest):
    """API xử lý đơn hàng"""
    try:
        # Pydantic already validated and parsed config_json and color_list
        config_parsed = data.config_json  # List[dict]
        color_list = data.color_list      # List[str]

        # 1. RAG Retrieval (safe if rag is test-mode)
        similar_cases = rag.find_similar_examples(data.user_input, n_results=3)
        
        # 2. Build Prompt
        prompt = assemble_prompt(
            config_parsed, 
            data.user_input, 
            similar_cases, 
            color_list
        )
        
        # 3. Call LLM (skip actual call in TESTING)
        if ai_client is None:
            # Return a deterministic dummy for tests or when TESTING=1
            return {
                "result": [],
                "used_examples": [ex['input'] for ex in similar_cases]
            }

        response = ai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.0 # Zero temperature để ổn định output
        )
        
        raw_result = response.choices[0].message.content
        
        # 4. Cleaning JSON Markdown
        clean_json = raw_result.replace("```json", "").replace("```", "").strip()
        parsed_result = json.loads(clean_json)
        
        return {
            "result": parsed_result,
            "used_examples": [ex['input'] for ex in similar_cases] # Metadata để debug
        }
        
    except ValueError as ve:
        # Catch Pydantic validation rethrows if any
        raise HTTPException(status_code=400, detail=str(ve))
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="LLM returned invalid JSON")
    except Exception as e:
        return {"error": str(e), "raw_response": locals().get('raw_result', '')}

# Lệnh chạy: uvicorn main:app --reload

# --- Root / UI ---
@app.get("/")
async def root():
    """Serve the main UI"""
    static_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(static_path):
        return FileResponse(static_path)
    return {"message": "Etsy RAG Agent API. Visit /docs for API documentation."}

# --- API Endpoints for Database Operations ---

# Training Examples API
class TrainingExampleCreate(BaseModel):
    user_input: str
    correct_output: str  # JSON string
    category: str = "general"

@app.get("/api/training-examples")
async def get_training_examples(category: Optional[str] = None, validated_only: bool = False):
    """Lấy danh sách training examples"""
    return db.get_training_examples(category=category, validated_only=validated_only)

@app.post("/api/training-examples")
async def create_training_example(data: TrainingExampleCreate):
    """Thêm training example mới"""
    try:
        # Validate JSON
        output_parsed = json.loads(data.correct_output)
        
        # Save to database
        example_id = db.add_training_example(
            user_input=data.user_input,
            correct_output=output_parsed,
            category=data.category
        )
        
        # Also add to RAG engine
        rag.add_example(data.user_input, data.correct_output, data.category)
        
        return {"status": "success", "id": example_id}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="correct_output must be valid JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/training-examples/{example_id}/validate")
async def validate_training_example(example_id: int):
    """Đánh dấu training example đã validate"""
    success = db.validate_training_example(example_id)
    if success:
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="Example not found")

@app.delete("/api/training-examples/{example_id}")
async def delete_training_example(example_id: int):
    """Xóa training example"""
    success = db.delete_training_example(example_id)
    if success:
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="Example not found")

# Config Fields API
class ConfigFieldCreate(BaseModel):
    name: str
    type: str
    options: Optional[dict] = None
    is_required: bool = True

@app.get("/api/config-fields")
async def get_config_fields():
    """Lấy danh sách config fields"""
    return db.get_config_fields()

@app.post("/api/config-fields")
async def create_config_field(data: ConfigFieldCreate):
    """Thêm config field mới"""
    try:
        field_id = db.add_config_field(
            name=data.name,
            field_type=data.type,
            options=data.options,
            is_required=data.is_required
        )
        return {"status": "success", "id": field_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/config-fields/{field_id}")
async def delete_config_field(field_id: int):
    """Xóa config field"""
    success = db.delete_config_field(field_id)
    if success:
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="Field not found")

# Colors API
class ColorCreate(BaseModel):
    name: str
    hex_code: Optional[str] = None

@app.get("/api/colors")
async def get_colors():
    """Lấy danh sách colors"""
    return db.get_colors()

@app.post("/api/colors")
async def create_color(data: ColorCreate):
    """Thêm color mới"""
    try:
        color_id = db.add_color(name=data.name, hex_code=data.hex_code)
        return {"status": "success", "id": color_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/colors/{color_id}")
async def delete_color(color_id: int):
    """Xóa color"""
    success = db.delete_color(color_id)
    if success:
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="Color not found")

# Stats and Validation API
@app.get("/api/stats")
async def get_stats():
    """Lấy thống kê về dữ liệu"""
    return db.get_validation_stats()

@app.get("/api/logs")
async def get_logs(limit: int = 100):
    """Lấy prediction logs"""
    return db.get_prediction_logs(limit=limit)