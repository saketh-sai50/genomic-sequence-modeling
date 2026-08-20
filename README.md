# 🧬 Genomic Sequence Modeling & RAG System

An AI-powered genomic sequence analysis system that combines deep learning, genomic sequence classification, Retrieval-Augmented Generation (RAG), vector search, and LLM-based explanations.

## 🚀 Overview

This project combines deep learning models for genomic sequence classification with a Retrieval-Augmented Generation (RAG) pipeline to provide context-aware genomic explanations.

The system uses trained sequence models to analyze DNA sequences and a RAG pipeline to retrieve relevant information from a genomic knowledge base before generating responses using Google's Gemini LLM.

## ✨ Key Features

- 🧬 Genomic DNA sequence preprocessing and analysis
- 🧠 RNN, LSTM, GRU and BiLSTM models
- 🔬 Genomic sequence classification
- 📚 Genomic knowledge base
- 🔎 Semantic retrieval using ChromaDB
- 🔗 Retrieval-Augmented Generation (RAG)
- 🤖 Gemini LLM for context-aware explanations
- 🌐 Flask-based web application

## 🏗️ Project Architecture

```text
                    User
                      │
                      ▼
             Flask Web Application
                      │
                      ▼
              DNA Sequence Input
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
   Deep Learning Model       RAG Pipeline
          │                       │
          ▼                       ▼
   Sequence Prediction      Knowledge Base
                                  │
                                  ▼
                             ChromaDB
                                  │
                                  ▼
                         Relevant Context
                                  │
                    ┌─────────────┘
                    ▼
               Gemini LLM
                    │
                    ▼
          Context-Aware Explanation
