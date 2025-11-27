# Etsy RAG Agent

Hệ thống xử lý đơn hàng Etsy thông minh sử dụng RAG (Retrieval-Augmented Generation) và AI.

## Tính năng

- 🤖 **AI Processing**: Sử dụng GPT-4 để phân tích và trích xuất thông tin từ đơn hàng
- 📚 **RAG Engine**: Lưu trữ và tìm kiếm các ví dụ tương tự để cải thiện kết quả
- 🗄️ **Database**: PostgreSQL để lưu trữ training data, config và logs
- 🎨 **Web UI**: Giao diện đầy đủ để nhập liệu và quản lý dữ liệu
- ✅ **Data Validation**: Kiểm tra dữ liệu đã đủ để hoạt động tốt

## Cài đặt

### Yêu cầu

- Docker & Docker Compose
- OpenAI API Key

### Bước 1: Clone và cấu hình

```bash
git clone <repo-url>
cd esty-rag-agent

# Copy và cấu hình environment
cp .env.example .env
# Sửa file .env, thêm OPENAI_API_KEY
```

### Bước 2: Chạy với Docker

```bash
# Build và chạy
docker-compose up -d

# Xem logs
docker-compose logs -f
```

### Bước 3: Truy cập

- **Web UI**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Sử dụng

### Web UI

1. Truy cập http://localhost:8000
2. **Training Data**: Thêm các ví dụ để AI học
3. **Config Fields**: Định nghĩa các trường dữ liệu cần trích xuất
4. **Colors**: Thêm danh sách màu sắc hỗ trợ
5. **Predict**: Test xử lý đơn hàng

### API Endpoints

#### Training Examples
```bash
# Thêm example
POST /api/training-examples
{
  "user_input": "Blue - 2022",
  "correct_output": "[{\"Color\": \"Blue\", \"Year\": \"2022\"}]",
  "category": "general"
}

# Lấy danh sách
GET /api/training-examples

# Validate example
POST /api/training-examples/{id}/validate

# Xóa example
DELETE /api/training-examples/{id}
```

#### Config Fields
```bash
# Thêm field
POST /api/config-fields
{
  "name": "Color",
  "type": "text",
  "is_required": true
}

# Lấy danh sách
GET /api/config-fields
```

#### Colors
```bash
# Thêm color
POST /api/colors
{
  "name": "Blue",
  "hex_code": "#0000FF"
}

# Lấy danh sách
GET /api/colors
```

#### Predict
```bash
POST /predict
{
  "user_input": "Blue - John 2022",
  "config_json": [{"name": "Color", "type": "text"}],
  "color_list": ["Blue", "Red", "Green"]
}
```

#### Statistics & Validation
```bash
# Lấy thống kê
GET /api/stats

# Response:
{
  "training_examples": {
    "total": 10,
    "validated": 5,
    "validation_rate": 50.0
  },
  "config_fields": 3,
  "colors": 15,
  "is_data_sufficient": true
}
```

## Kiểm tra Data Validation

Hệ thống kiểm tra các điều kiện sau để xác định dữ liệu đã đủ:

- ✅ Tối thiểu 10 training examples
- ✅ Tối thiểu 5 examples đã được validate
- ✅ Có ít nhất 1 config field
- ✅ Có ít nhất 1 color

Trạng thái validation hiển thị trên dashboard với các warning cụ thể nếu chưa đủ dữ liệu.

## Cấu trúc Project

```
esty-rag-agent/
├── main.py              # FastAPI app và API endpoints
├── rag_engine.py        # RAG engine với ChromaDB
├── prompt_manager.py    # Xây dựng prompt cho LLM
├── database.py          # Database models và operations
├── static/
│   └── index.html       # Web UI
├── docker-compose.yml   # Docker config với PostgreSQL
├── Dockerfile           # Docker image config
├── init.sql             # Database initialization
├── requirements.txt     # Python dependencies
└── tests/               # Unit tests
```

## Development

### Chạy local (không Docker)

```bash
# Cài dependencies
pip install -r requirements.txt

# Chạy với SQLite (test mode)
TESTING=1 uvicorn main:app --reload

# Chạy với PostgreSQL
DATABASE_URL=postgresql://user:pass@localhost:5432/db uvicorn main:app --reload
```

### Chạy tests

```bash
TESTING=1 pytest tests/ -v
```

## License

MIT
