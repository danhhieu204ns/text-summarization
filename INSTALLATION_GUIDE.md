# Hướng Dẫn Cài Đặt Dự Án Text Summarization Demo

## Tổng Quan
Dự án này là một ứng dụng tóm tắt văn bản tiếng Việt sử dụng:
- **Backend**: FastAPI với các mô hình PhoBERT (extractive) và ViT5 với LoRA (abstractive)
- **Frontend**: React + Vite

## Yêu Cầu Hệ Thống

### Phần Mềm Cần Thiết
- **Python**: 3.8 trở lên (khuyến nghị Python 3.10)
- **Node.js**: 16.x trở lên
- **npm**: 8.x trở lên
- **Git**: để clone repository
- **CUDA** (tùy chọn): để sử dụng GPU acceleration

### Phần Cứng Khuyến Nghị
- **RAM**: Tối thiểu 8GB (khuyến nghị 16GB)
- **GPU**: NVIDIA GPU với CUDA hỗ trợ (tùy chọn, nhưng cải thiện hiệu suất đáng kể)
- **Dung lượng ổ đĩa**: Tối thiểu 5GB cho models và dependencies

---

## Cài Đặt

### Bước 1: Clone/Tải Dự Án
```bash
# Nếu dùng Git
git clone <repository-url>
cd nlp

# Hoặc giải nén file zip vào thư mục nlp
```

### Bước 2: Cài Đặt Backend (FastAPI)

#### 2.1. Tạo Môi Trường Ảo Python
```powershell
# Di chuyển vào thư mục backend
cd summarization-demo\backend

# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Nếu gặp lỗi execution policy, chạy lệnh này:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### 2.2. Cài Đặt Dependencies
```powershell
# Cài đặt các thư viện từ requirements.txt
pip install -r requirements.txt
```

**Lưu ý**: 
- Nếu có GPU NVIDIA, PyTorch sẽ tự động sử dụng CUDA
- Quá trình cài đặt có thể mất 5-10 phút tùy theo tốc độ internet

#### 2.3. Kiểm Tra Đường Dẫn Model
Đảm bảo file `app.py` có đường dẫn đúng đến LoRA adapter:
```python
lora_adapter_path = r"D:\WORKSPACE\nlp\results\t5_new_lora\checkpoint-1865"
```

**Cập nhật đường dẫn này** nếu thư mục dự án của bạn ở vị trí khác. Ví dụ:
```python
lora_adapter_path = r"C:\Users\YourName\Projects\nlp\results\t5_new_lora\checkpoint-1865"
```

### Bước 3: Cài Đặt Frontend (React)

#### 3.1. Di Chuyển Vào Thư Mục Frontend
```powershell
# Từ thư mục gốc nlp
cd summarization-demo\frontend
```

#### 3.2. Cài Đặt Dependencies
```powershell
# Cài đặt các package từ package.json
npm install
```

**Lưu ý**: Quá trình này có thể mất 2-5 phút

---

## Chạy Ứng Dụng

### Chạy Backend

```powershell
# Trong thư mục backend với môi trường ảo đã kích hoạt
cd summarization-demo\backend
.\venv\Scripts\Activate.ps1  # Nếu chưa kích hoạt
uvicorn app:app --reload
```

Backend sẽ chạy tại: **http://localhost:8000**

Kiểm tra API docs tại: **http://localhost:8000/docs**

### Chạy Frontend

Mở một **terminal mới** (không tắt terminal backend):

```powershell
# Trong thư mục frontend
cd summarization-demo\frontend
npm run dev
```

Frontend sẽ chạy tại: **http://localhost:5173** (hoặc port khác nếu 5173 đã được sử dụng)

---

## Sử Dụng Ứng Dụng

1. Mở trình duyệt và truy cập: **http://localhost:5173**
2. Nhập văn bản tiếng Việt cần tóm tắt
3. Chọn chế độ tóm tắt:
   - **Extractive**: Trích xuất các câu quan trọng từ văn bản gốc
   - **Abstractive**: Tạo tóm tắt mới bằng cách paraphrase
   - **Both**: Hiển thị cả hai kết quả
4. Nhấn "Summarize" để xem kết quả

---

## Xử Lý Sự Cố

### Lỗi Import hoặc Module Not Found
```powershell
# Đảm bảo môi trường ảo đã được kích hoạt
.\venv\Scripts\Activate.ps1

# Cài đặt lại requirements
pip install -r requirements.txt
```

### Lỗi CUDA/GPU
Nếu bạn gặp lỗi liên quan đến CUDA nhưng không có GPU:
- Ứng dụng sẽ tự động chuyển sang sử dụng CPU
- Thời gian xử lý sẽ chậm hơn nhưng vẫn hoạt động

### Lỗi Port Already in Use

**Backend (port 8000)**:
```powershell
# Sử dụng port khác
uvicorn app:app --reload --port 8001
```

**Frontend**: Vite sẽ tự động chọn port khác nếu 5173 đã được sử dụng

### Lỗi CORS
Nếu frontend không kết nối được với backend:
1. Kiểm tra backend đang chạy tại http://localhost:8000
2. Kiểm tra trong `app.py` đã có cấu hình CORS:
   ```python
   allow_origins=["*"]
   ```

### Lỗi Model Path Not Found
```powershell
# Kiểm tra đường dẫn model có tồn tại
Test-Path "D:\WORKSPACE\nlp\results\t5_new_lora\checkpoint-1865"

# Nếu False, cập nhật lại đường dẫn trong app.py
```

### Frontend Build Fails
```powershell
# Xóa node_modules và cài lại
Remove-Item -Recurse -Force node_modules
npm install
```

---

## Cấu Trúc Dự Án

```
nlp/
├── summarization-demo/
│   ├── backend/
│   │   ├── app.py                 # FastAPI application
│   │   ├── requirements.txt       # Python dependencies
│   │   └── venv/                  # Python virtual environment
│   └── frontend/
│       ├── src/
│       │   ├── App.jsx           # React main component
│       │   └── ...
│       ├── package.json          # Node.js dependencies
│       └── node_modules/         # Node.js packages
├── results/
│   └── t5_new_lora/
│       └── checkpoint-1865/      # LoRA adapter cho ViT5
└── requirements.txt              # Root Python dependencies
```

---

## API Endpoints

### POST /summarize
Tóm tắt văn bản

**Request Body**:
```json
{
  "text": "Văn bản cần tóm tắt...",
  "mode": "both",
  "top_k": 3
}
```

**Parameters**:
- `text` (string, required): Văn bản đầu vào
- `mode` (string, required): "extractive", "abstractive", hoặc "both"
- `top_k` (integer, optional): Số câu cho extractive summarization

**Response**:
```json
{
  "extractive": "Tóm tắt extractive...",
  "abstractive": "Tóm tắt abstractive..."
}
```

### GET /
Health check endpoint

---

## Ghi Chú Bổ Sung

### Tối Ưu Hiệu Suất
- Sử dụng GPU nếu có thể để tăng tốc độ xử lý
- Với văn bản dài, thời gian xử lý sẽ lâu hơn
- Extractive summarization nhanh hơn abstractive

### Giới Hạn input
- Độ dài văn bản tối đa: 1024 tokens cho abstractive
- Extractive không có giới hạn cứng nhưng khuyến nghị < 2000 từ

### Models Sử Dụng
- **PhoBERT** (vinai/phobert-base): Pre-trained Vietnamese BERT
- **ViT5** (VietAI/vit5-base): Vietnamese T5 with LoRA fine-tuning
- **Underthesea**: Vietnamese NLP toolkit cho tokenization

---

## Liên Hệ và Hỗ Trợ

Nếu gặp vấn đề trong quá trình cài đặt:
1. Kiểm tra lại các bước cài đặt
2. Đảm bảo đã cài đặt đúng phiên bản Python và Node.js
3. Kiểm tra logs trong terminal để xác định lỗi cụ thể

---

## License

Dự án này sử dụng các mô hình open-source:
- PhoBERT: MIT License
- ViT5: MIT License
- FastAPI: MIT License
- React: MIT License
