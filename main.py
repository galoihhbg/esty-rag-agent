from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List, Union, Optional, Any
from rag_engine import RagEngine
from prompt_manager import assemble_prompt
from openai import OpenAI
import os
import json

# Khởi tạo App
app = FastAPI(title="Etsy RAG Agent")

# Lazy/conditional initialization to make tests easier
if os.getenv("TESTING", "0") != "1":
    rag = RagEngine()
    ai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
else:
    rag = RagEngine(test_mode=True)
    ai_client = None

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