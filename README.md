# 🏥 HouseGPT – Medical-Diagnostic-Agent-Multimodal

[![Python Version](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/) 
[![Poetry](https://img.shields.io/badge/Poetry-1.5.1-blueviolet)](https://python-poetry.org/) 
[![Qdrant](https://img.shields.io/badge/Qdrant-vectorDB-orange)](https://qdrant.io/) 
[![Ollama](https://img.shields.io/badge/Ollama-LLM-purple)](https://ollama.com/) 
[![LangGraph](https://img.shields.io/badge/LangGraph-GraphLLM-brightgreen)](https://github.com/langgraph/langgraph)
[![Qwen](https://img.shields.io/badge/Qwen-LLM-blueviolet)](https://huggingface.co/qwen) 
[![Whisper](https://img.shields.io/badge/Whisper-ASR-red)](https://github.com/openai/whisper)

---

## Overview

**HouseGPT** is an advanced AI assistant designed to mimic the **diagnostic reasoning of Dr. House**.  
It combines **language understanding, multimodal processing, and long-term memory** to provide **context-aware, expert-level medical support**.

Key aspects:

- **RAG-powered knowledge retrieval** using Qdrant for vectorized memory
- **Node-based LangGraph architecture** for flexible reasoning and routing
- **Multimodal inputs and outputs** (text, image, audio)
- **Ollama embeddings** for semantic understanding and memory operations

This system is built for professional-grade AI workflows, suitable for **research, experimentation, and production**.

---

## Features

#### Diagnostic & Reasoning AI

- Contextual conversation with memory injection
- Contextual responses grounded in patient history and medical knowledge
- Structured outputs using Pydantic schemas
- Router logic for intelligent reasoning chains

#### Retrieval-Augmented Generation (RAG)

- Qdrant vector store for storing and retrieving medical knowledge
- Semantic search for relevant documents and past cases
- Memory scoring to highlight important knowledge for reasoning

#### Multimodal Capabilities

- **Text-to-Image** for visual medical scenarios
- **Image-to-Text** for analyzing scans or medical images
- **Audio-to-Text** transcription for voice inputs
- **Text-to-Audio** synthesis for outputs

#### Core Technologies

- **Python 3.11**
- **Poetry** dependency management
- **FastAPI** backend for API endpoints
- **Qdrant** for vector storage
- **LangGraph** for modular reasoning and node orchestration
- **Ollama embeddings** for semantic memory
- **Qwen3** for text conversation and image to text
- **Whisper** for text to audio

---

## Architecture

This layout ensures:

- Scalability for multi-agent systems
- Clear separation of concerns
- Easy integration of new LLM models or multimodal nodes

---

## Installation

- Clone the repo:

```bash
git clone https://github.com/Far3s18/HouseGPT---Medical-Diagnostic-Agent-Multimodal.git
cd HouseGPT---Medical-Diagnostic-Agent-Multimodal
```

- Install dependencies via Poetry:

```bash
poetry install
poetry shell
```

- Create your environment variables:

```
cp .env.example .env
```
---

## Configuration

The project expects environment variables for:

1. Qdrant endpoint and API key (for vector memory)
2. Ollama model configs (for LLM embeddings & reasoning)
3. Optional model providers such as Qwen or Whisper
4. API keys if using external LLM services

---

## API Usage

Start the FastAPI server:

```
uvicorn app.main:app --reload
```


### Endpoints


| Endpoint                       | Method | Description                                    |
|--------------------------------|--------|------------------------------------------------|
| `/api/v1/chat`                  | POST   | Send a text message and receive AI reasoning  |
| `/api/v1/memory`                | POST   | Retrieve or store contextual memory           |
| `/api/v1/multimodal/text`       | POST   | Generate image or audio from text input       |
| `/api/v1/multimodal/image`      | POST   | Convert image input into text for analysis    |
| `/api/v1/multimodal/audio`      | POST   | Transcribe audio input to text                |

---

## Roadmap & Future Work

- Add some tools (e.g: search web, deep thinking... etc)
- Expanded multimodal reasoning
- Advanced memory summarization
- Additional clinical knowledge sources
- Improved interpreter chains and model adapters
