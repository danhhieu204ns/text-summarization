# 📝 Dự Án Tóm Tắt Văn Bản Tiếng Việt (Vietnamese Text Summarization)

## Tổng Quan

Dự án nghiên cứu và triển khai hai phương pháp tóm tắt văn bản tiếng Việt dựa trên Deep Learning:
- **Extractive Summarization**: Trích xuất các câu quan trọng nhất sử dụng PhoBERT + MMR
- **Abstractive Summarization**: Tạo tóm tắt mới bằng ViT5 với Parameter-Efficient Fine-Tuning (LoRA)

Dự án bao gồm các Jupyter notebooks để huấn luyện, đánh giá models, và một web demo đơn giản (FastAPI + React) để trải nghiệm kết quả.

### Đặc Điểm Chính

- **Hai phương pháp AI**: Extractive (PhoBERT + MMR) và Abstractive (ViT5 + LoRA)
- **Tối ưu cho tiếng Việt**: Sử dụng PhoBERT và ViT5 - các models cho tiếng Việt
- **Parameter-Efficient Fine-Tuning**: Sử dụng LoRA thay vì full fine-tuning để tiết kiệm tài nguyên
- **Notebooks chi tiết**: Từ data preprocessing, training, đến evaluation
- **Web demo**: Giao diện đơn giản để test models

## Kiến Trúc AI

```
INPUT TEXT (Vietnamese)
         │
    ┌────┴──────────────────────────┐
    │                               │
    ▼                               ▼
┌─────────────────┐         ┌─────────────────┐
│  EXTRACTIVE     │         │  ABSTRACTIVE    │
│  PhoBERT-base   │         │  ViT5-base      │
│  + MMR          │         │  + LoRA         │
└─────────────────┘         └─────────────────┘
    │                               │
    │ Sentence                      │ Seq2Seq
    │ Selection                     │ Generation
    │                               │
    ▼                               ▼
Selected Sentences          Generated Summary
```

## Công Nghệ và Models

### Core AI Components
- **PyTorch 2.1+**: Deep learning framework
- **Transformers (Hugging Face)**: Pre-trained models và tokenizers
- **PEFT (LoRA)**: Parameter-Efficient Fine-Tuning - giảm parameters cần train 99%
- **Underthesea**: Vietnamese NLP toolkit cho tokenization

### Pre-trained Models

#### 1. PhoBERT-base (vinai/phobert-base)
- **Kiến trúc**: RoBERTa-base adapted for Vietnamese
- **Parameters**: 135M
- **Pre-training**: 20GB Vietnamese text (Wikipedia, news, social media)
- **Sử dụng**: Tạo sentence embeddings cho extractive summarization

#### 2. ViT5-base (VietAI/vit5-base)
- **Kiến trúc**: T5 (Text-to-Text Transfer Transformer) for Vietnamese
- **Parameters**: 250M (base model) + ~0.6M (LoRA adapter)
- **Pre-training**: Vietnamese Wikipedia, news, books
- **Fine-tuning**: Sử dụng LoRA trên dataset tóm tắt tin tức tiếng Việt

### Algorithms

#### Maximal Marginal Relevance (MMR)
- **Công thức**: `MMR = λ × Sim(Si, D) - (1-λ) × max Sim(Si, Sj)`
- **Mục đích**: Balance giữa relevance và diversity
- **Tham số λ**: Điều chỉnh trade-off (λ=0.7 trong dự án)

### Web Demo (FastAPI + React)
Một giao diện đơn giản để test các models đã huấn luyện:
- **Backend**: FastAPI phục vụ inference API
- **Frontend**: React UI để input/output
- **Mục đích**: Demo và validation

## Yêu Cầu Hệ Thống

### Phần Mềm
- Python 3.8+ (khuyến nghị 3.10)
- Jupyter Notebook / JupyterLab
- CUDA toolkit 11.8+ (cho GPU training/inference)

### Phần Cứng
- **RAM**: Tối thiểu 16GB (khuyến nghị 32GB cho training)
- **GPU**: NVIDIA GPU với ít nhất 8GB VRAM (khuyến nghị RTX 3060 trở lên)
- **Ổ đĩa**: 10GB+ cho models, datasets, và checkpoints

### Cho Web Demo (tùy chọn)
- Node.js 16+ và npm 8+ (nếu muốn chạy React frontend)
- RAM 8GB+ là đủ cho inference

## Hướng Dẫn Sử Dụng

### 1. Setup Môi Trường

```powershell
# Clone dự án
git clone <repository-url>
cd nlp

# Tạo môi trường ảo Python
python -m venv venv
.\venv\Scripts\Activate.ps1

# Cài đặt dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers peft underthesea scikit-learn numpy pandas jupyter
```

### 2. Khám Phá Notebooks

#### `extractive-summarization.ipynb`
Notebook thử nghiệm extractive summarization:
- Load PhoBERT và tạo sentence embeddings
- Implement và so sánh các thuật toán: MMR, TextRank, centroid-based
- Đánh giá kết quả trên văn bản mẫu
- Visualization similarity matrix

#### `abstractive-summarization.ipynb`
Notebook huấn luyện abstractive model:
- Load và preprocess dataset tóm tắt tin tức
- Fine-tune ViT5 với LoRA adapter
- Training loop với validation
- Evaluation metrics (ROUGE scores)
- Generate và so sánh kết quả
- Save checkpoint để deploy

### 3. Chạy Web Demo (Tùy Chọn)

Nếu muốn test qua giao diện web:

```powershell
# Backend
cd summarization-demo\backend
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --reload

# Frontend (terminal mới)
cd summarization-demo\frontend
npm install
npm run dev
```

Truy cập http://localhost:5173 để test models.

## API Reference (Web Demo)

Web demo cung cấp REST API đơn giản để inference:

### POST `/summarize`

**Request:**
```json
{
  "text": "Văn bản cần tóm tắt...",
  "mode": "both",
  "top_k": 3
}
```

**Response:**
```json
{
  "extractive": "Câu 1. Câu 2. Câu 3.",
  "abstractive": "Tóm tắt được generate..."
}
```

Chi tiết xem [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md).

## Cấu Trúc Dự Án

```
nlp/
├── abstractive-summarization.ipynb    # [CORE] Notebook train ViT5 + LoRA
│       ├── Data loading & preprocessing
│       ├── LoRA configuration
│       ├── Training loop với validation
│       ├── Evaluation (ROUGE scores)
│       └── Save checkpoint
│
├── extractive-summarization.ipynb     # [CORE] Notebook thử nghiệm extractive
│       ├── PhoBERT embedding
│       ├── MMR algorithm implementation
│       ├── Comparison với TextRank, centroid
│       └── Visualization
│
├── documents.txt                       # Sample data để test
├── INSTALLATION_GUIDE.md               # Hướng dẫn setup chi tiết
├── README.md                           # File này
│
├── results/
│   └── t5_new_lora/
│       └── checkpoint-1865/               # [CORE] LoRA adapter trained
│           ├── adapter_config.json        # LoRA hyperparameters
│           ├── adapter_model.safetensors  # Trained weights (~2.5MB)
│           ├── trainer_state.json         # Training history
│           └── tokenizer files...
│
└── summarization-demo/                 # [DEMO] Web application
    ├── backend/                        # FastAPI server
    │   ├── app.py                      # Load models + API endpoints
    │   └── requirements.txt
    └── frontend/                       # React UI
        ├── src/App.jsx                 # Simple UI
        └── package.json
```

## Phương Pháp AI

### 1. Extractive Summarization: PhoBERT + MMR

#### Kiến Trúc và Quy Trình

```
Input Text
    │
    ├─► Sentence Tokenization (Underthesea)
    │
    ├─► PhoBERT Encoding
    │   └─► [CLS] token embedding cho mỗi câu
    │       └─► 768-dim vectors
    │
    ├─► Compute Similarities
    │   ├─► Sentence ↔ Document (relevance)
    │   └─► Sentence ↔ Selected Sentences (diversity)
    │
    ├─► MMR Selection Algorithm
    │   └─► Iteratively chọn câu tối ưu
    │       MMR = λ·Sim(Si,D) - (1-λ)·max(Sim(Si,Sj))
    │
    └─► Output: Top-k sentences (sorted by position)
```

### 2. Abstractive Summarization: ViT5 + LoRA

#### Kiến Trúc và Quy Trình

```
Input Text
    │
    ├─► Preprocessing: "summarize: " + text
    │
    ├─► ViT5 Tokenizer
    │   └─► Input IDs [batch, seq_len]
    │
    ├─► ViT5 Encoder (frozen base weights)
    │   └─► Contextualized embeddings
    │
    ├─► LoRA Adapter Layers (trainable)
    │   └─► Low-rank matrices: A [r×d], B [d×r]
    │   └─► Updated weights: W' = W + B·A
    │
    ├─► ViT5 Decoder (frozen base + LoRA)
    │   └─► Autoregressive generation
    │
    └─► Output: Generated summary
```

#### Parameter-Efficient Fine-Tuning với LoRA

**Ý tưởng chính:**
- Full fine-tuning: Update tất cả 250M parameters
- LoRA: Chỉ train ~0.6M parameters (0.24%) bằng cách:
  - Freeze tất cả weights của base model
  - Thêm low-rank adapter matrices vào attention layers

**LoRA Math:**
```
Original attention: h = W₀·x
LoRA update: h = W₀·x + ΔW·x
           where ΔW = B·A
           
W₀: [d×d] frozen weights
A: [r×d] trainable (down-projection)  
B: [d×r] trainable (up-projection)
r: rank (8 hoặc 16) << d (768)

Parameters to train: 2·r·d thay vì d²
```

## Cấu Hình và Tùy Chỉnh

### Extractive Hyperparameters

```python
# Trong notebook hoặc app.py

# MMR parameters
lambda_ = 0.7              # Balance: 0.7 relevance + 0.3 diversity
                           # Tăng λ → more relevant, less diverse
                           # Giảm λ → more diverse, less relevant

top_k = 3                  # Số câu trong tóm tắt
                           # Auto: dựa vào avg sentence length
                           # Manual: user specify

# PhoBERT encoding
max_length = 256           # Max tokens per sentence
                           # Câu dài hơn sẽ bị truncate
```

## Performance và Benchmarks

### Quality Metrics (trên test set)

#### Abstractive (ViT5 + LoRA)
- **ROUGE-1**: 53.72
- **ROUGE-2**: 26.23  
- **ROUGE-L**: 36.17
- **ROUGE-Lsum**: 36.20
- **BLEU (sacreBLEU)**: 8.45
- **BERTScore**:
  - Precision: 0.70
  - Recall: 0.70
  - F1: 0.70

#### Extractive (PhoBERT + MMR)
- **ROUGE-1**: 37.79
- **ROUGE-2**: 18.25
- **ROUGE-L**: 24.70
- **ROUGE-Lsum**: 24.83
- **BLEU**: 3.65
- **BERTScore**:
  - Precision: 0.65
  - Recall: 0.73
  - F1: 0.69

*ROUGE scores càng cao (max=100.0) càng giống reference summary*
*BERTScore đo semantic similarity, F1 càng cao càng tốt (max=1.0)*


## References và Papers

### Foundational Papers

1. **BERT & RoBERTa**
   - Devlin et al. (2018): "BERT: Pre-training of Deep Bidirectional Transformers"
   - Liu et al. (2019): "RoBERTa: A Robustly Optimized BERT Pretraining Approach"

2. **T5 Architecture**
   - Raffel et al. (2020): "Exploring the Limits of Transfer Learning with T5"
   - Text-to-Text framework cho NLP tasks

3. **LoRA (Low-Rank Adaptation)**
   - Hu et al. (2021): "LoRA: Low-Rank Adaptation of Large Language Models"
   - ICLR 2022, Microsoft Research
   - Key insight: `ΔW = B·A` với rank r << d

4. **MMR (Maximal Marginal Relevance)**
   - Carbonell & Goldstein (1998): "The Use of MMR for Text Summarization"
   - Balance relevance và diversity

### Vietnamese NLP Models

1. **PhoBERT**
   - Nguyen & Nguyen (2020): "PhoBERT: Pre-trained language models for Vietnamese"
   - VinAI Research
   - Paper: https://arxiv.org/abs/2003.00744

2. **ViT5**
   - VietAI Team (2021): "ViT5: Pretrained Text-to-Text Transformer for Vietnamese"
   - Based on T5 architecture
   - GitHub: https://github.com/vietai/ViT5

### Datasets

- OpenHust/vietnamese-summarization 


**⭐ Nếu dự án hữu ích cho research/learning, hãy cho một star! ⭐**
