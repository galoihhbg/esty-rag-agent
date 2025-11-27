"""
Prompt Manager - Module để xây dựng prompt cho LLM
"""
import json
from typing import List, Dict, Any


def assemble_prompt(
    config_fields: List[Dict[str, Any]],
    user_input: str,
    similar_cases: List[Dict[str, Any]],
    color_list: List[str]
) -> str:
    """
    Xây dựng prompt để gửi cho LLM.
    
    Args:
        config_fields: Danh sách các trường cấu hình (name, type, options)
        user_input: Input từ người dùng cần xử lý
        similar_cases: Các case tương tự từ RAG
        color_list: Danh sách màu sắc được hỗ trợ
        
    Returns:
        str: Prompt đầy đủ cho LLM
    """
    
    # Build field descriptions
    field_descriptions = []
    for field in config_fields:
        field_desc = f"- {field['name']} (type: {field['type']})"
        if field.get('options'):
            field_desc += f" - options: {json.dumps(field['options'])}"
        field_descriptions.append(field_desc)
    
    fields_text = "\n".join(field_descriptions)
    
    # Build similar cases examples
    examples_text = ""
    if similar_cases:
        examples_text = "\n\n**Các ví dụ tương tự:**\n"
        for i, case in enumerate(similar_cases, 1):
            examples_text += f"\nVí dụ {i}:\n"
            examples_text += f"Input: {case.get('input', '')}\n"
            examples_text += f"Output: {case.get('output', '')}\n"
    
    # Build color list
    colors_text = ", ".join(color_list) if color_list else "Không có màu sắc được chỉ định"
    
    prompt = f"""Bạn là AI assistant chuyên xử lý đơn hàng Etsy. Nhiệm vụ của bạn là phân tích input từ khách hàng và trích xuất thông tin thành JSON.

**Các trường dữ liệu cần trích xuất:**
{fields_text}

**Danh sách màu sắc hỗ trợ:** {colors_text}

{examples_text}

**Input cần xử lý:**
{user_input}

**Yêu cầu:**
1. Phân tích input và trích xuất thông tin theo các trường được định nghĩa
2. Nếu có màu sắc, map về màu gần nhất trong danh sách hỗ trợ
3. Trả về kết quả dạng JSON array
4. Chỉ trả về JSON, không kèm giải thích

**Output (JSON only):**"""
    
    return prompt


def validate_output_against_config(
    output: List[Dict[str, Any]],
    config_fields: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Validate output JSON có đúng cấu trúc theo config không.
    
    Args:
        output: Output JSON từ LLM
        config_fields: Config định nghĩa các trường
        
    Returns:
        Dict với keys: valid (bool), errors (list), warnings (list)
    """
    errors = []
    warnings = []
    
    if not isinstance(output, list):
        errors.append("Output phải là một JSON array")
        return {"valid": False, "errors": errors, "warnings": warnings}
    
    field_names = {f['name'] for f in config_fields}
    
    for idx, item in enumerate(output):
        if not isinstance(item, dict):
            errors.append(f"Item {idx} phải là một object")
            continue
            
        # Check for unknown fields
        item_fields = set(item.keys())
        unknown = item_fields - field_names
        if unknown:
            warnings.append(f"Item {idx} có các trường không xác định: {unknown}")
        
        # Check for missing required fields
        missing = field_names - item_fields
        if missing:
            warnings.append(f"Item {idx} thiếu các trường: {missing}")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }
