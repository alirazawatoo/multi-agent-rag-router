# PWB AI Agent - Multi-Agent RAG System

A proof-of-concept agentic AI system built with **LangGraph** for multi-agent orchestration, featuring retrieval-augmented generation (RAG) over curated documents with intelligent routing, retrieval, composition, and critique capabilities.

## 🎯 Overview

This POC demonstrates a sophisticated multi-agent architecture that can:
- **Route queries** intelligently between knowledge retrieval and action execution
- **Retrieve relevant context** from a vector database using semantic search
- **Generate detailed responses** with inline citations using Groq's fast inference
- **Critique and validate** responses for proper grounding
- **Execute safe actions** with audit logging (pause/unpause items, update hours)
- **Provide real-time decision tracking** through an interactive Streamlit UI

## 🏗️ Architecture

### Multi-Agent Pipeline (LangGraph)
```
Query → Router → [Knowledge Path | Action Path]
                      ↓              ↓
                  Retriever      Action Parser
                      ↓              ↓
                   Compose         END
                      ↓
                   Critic
                      ↓
                    END
```

### Core Components
- **Router Agent**: Classifies queries as knowledge requests or action commands
- **Retriever Agent**: Performs semantic search over document embeddings
- **Compose Agent**: Generates responses with citations using Groq LLM
- **Critic Agent**: Validates response grounding and citation quality
- **Action Agent**: Parses and proposes safe operational actions

## 🚀 Features

### Knowledge Retrieval
- **Vector Search**: Powered by `sentence-transformers` (bge-small-en-v1.5) + ChromaDB
- **Semantic Chunking**: Intelligent document processing with metadata preservation
- **Citation System**: Automatic inline citations with source tracking
- **Configurable Retrieval**: Adjustable top-k results and response styles

### Action Execution
- **Safe Operations**: Pause/unpause items, update operating hours
- **Approval Workflow**: Human-in-the-loop validation before execution
- **Audit Logging**: Complete action history with operator tracking
- **API Integration**: RESTful connector for operational systems

### Interactive UI
- **ChatGPT-style Interface**: Natural conversation flow with Streamlit
- **Decision Log**: Real-time visibility into agent reasoning
- **Source Display**: Retrieved documents with relevance scores
- **Parameter Controls**: Temperature, top-k, response style adjustment

## 📁 Project Structure

```
PWB_AI_Agent/
├── agents/
│   ├── graph.py          # LangGraph multi-agent orchestration
│   ├── llm.py           # Groq LLM integration with streaming
│   └── __init__.py
├── apps/ui/
│   └── streamlit_app.py # Interactive web interface
├── connectors/
│   └── ops_stub_api/    # Operational API connector (demo)
├── ingestion/
│   ├── build_index.py   # Document ingestion pipeline
│   └── chunking.py      # Semantic chunking utilities
├── data/
│   └── raw/             # Source documents (PDFs)
├── scripts/
│   ├── generate_pdfs.py # Sample document generator
│   └── query_index.py   # CLI query interface
├── requirements.txt     # Python dependencies
├── .env.example        # Environment configuration template
└── README.md           # This file
```

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.9+
- Git

### 1. Clone Repository
```bash
git clone https://github.com/thecodeman001/PWB_AI_Agent.git
cd PWB_AI_Agent
```

### 2. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
# Required: GROQ_API_KEY for LLM inference
# Optional: Customize paths and model settings
```

### 4. Document Ingestion
```bash
# Add your PDFs to data/raw/
# Then build the vector index
python ingestion/build_index.py
```

### 5. Launch Application
```bash
# Start Streamlit UI
streamlit run apps/ui/streamlit_app.py

# Or use CLI interface
python scripts/query_index.py "Your question here"
```

## 🔧 Configuration

### Environment Variables (.env)
```bash
# Vector Database
CHROMA_DB_DIR=data/chroma
RAW_DOCS_DIR=data/raw
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# LLM Provider
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b

# Optional: Action API
OPS_API_URL=http://localhost:8001
OPERATOR_NAME=demo_user
```

### Response Styles
- **Detailed**: Comprehensive responses with full context
- **Concise**: Brief, focused answers

### Supported Actions
- `pause "item name"` - Pause operational item
- `unpause "item name"` - Resume operational item  
- `update hours [details]` - Modify operating hours

## 📊 Usage Examples

### Knowledge Queries
```
"What are the onboarding requirements for new employees?"
"Explain the financial reporting process"
"How do we handle platform integrations?"
```

### Action Commands
```
"pause 'daily backup process'"
"update hours to 9 AM - 6 PM Monday through Friday"
"unpause the customer service queue"
```

## 🔍 Technical Details

### Dependencies
- **LangGraph 0.2.39**: Multi-agent orchestration framework
- **ChromaDB 0.5.5**: Vector database for embeddings
- **Sentence-Transformers 2.7.0**: Embedding model inference
- **Streamlit 1.38.0**: Interactive web interface
- **Groq 0.11.0**: Fast LLM inference API
- **FastAPI 0.115.6**: Action API framework

### Performance
- **Embedding Model**: BGE-small-en-v1.5 (384 dimensions)
- **Vector Search**: Cosine similarity with configurable top-k
- **LLM Inference**: Groq's optimized infrastructure
- **Streaming**: Real-time response generation

## 🚦 Development

### Adding New Agents
1. Define agent function in `agents/graph.py`
2. Add to StateGraph with appropriate edges
3. Update routing logic if needed

### Custom Document Processing
1. Modify `ingestion/chunking.py` for new formats
2. Update metadata extraction in `build_index.py`
3. Rebuild vector index

### API Extensions
1. Add endpoints to `connectors/ops_stub_api/main.py`
2. Update action parsing in `agents/graph.py`
3. Extend UI approval workflow

## 📝 License

This project is open-source and available under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For questions or issues:
- Open a GitHub issue
- Review the decision logs in the UI for debugging
- Check environment configuration

---

**Built with ❤️ for intelligent document processing and operational automation**
