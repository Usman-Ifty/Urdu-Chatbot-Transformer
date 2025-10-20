🗣️ Urdu Chatbot — Transformer with Multi-Head Attention

A Transformer-based Urdu Conversational Chatbot built completely from scratch using PyTorch, trained on an Urdu dialogue dataset and deployed with a Streamlit web interface for real-time interaction.

This project demonstrates an end-to-end Natural Language Processing (NLP) pipeline — from text normalization and tokenization, to training, evaluation, and web deployment — optimized for Urdu text generation.

🧠 Project Overview

The goal of this project is to design and implement a sequence-to-sequence Transformer capable of understanding and generating coherent Urdu sentences without relying on pre-trained language models.

It uses:

Encoder-Decoder Transformer architecture

Multi-Head Attention for contextual representation

SentencePiece Tokenization for subword-level Urdu handling

Beam Search & Greedy decoding

Streamlit Interface for live Urdu conversation

🚀 Key Features

✅ Built from scratch using PyTorch
✅ Handles Urdu text (right-to-left) using Unicode
✅ Custom SentencePiece tokenizer (urdu_tokenizer.model)
✅ Real-time chat via Streamlit Web App
✅ Multiple decoding options (Beam Search / Greedy)
✅ Automatic masking & positional encoding
✅ Modular Transformer implementation (Encoder + Decoder)
✅ Supports state_dict checkpoint loading (urdu_chatbot.pt)

🏗️ Architecture Summary
Component	Description
Embedding Dim	128
Encoder Layers	2
Decoder Layers	2
Attention Heads	4
Feed-Forward Dim	512
Dropout	0.2
BOS / EOS IDs	2 / 3
Optimizer	Adam (lr = 1e-4)
Loss Function	Cross Entropy
📂 Repository Structure
Urdu-Chatbot-Transformer/
│
├── app.py                   # Streamlit chatbot app (inference)
├── urdu_chatbot.pt          # Trained model checkpoint (state_dict)
├── urdu_tokenizer.model     # SentencePiece tokenizer
├── normalized_sentences.csv # Urdu conversational dataset
├── nlpass2.ipynb            # Model training notebook
└── README.md                # Project documentation

🧰 Requirements

Install required packages before running:

pip install torch streamlit sentencepiece sacrebleu rouge-score


(Optionally, you can create a requirements.txt with the same list.)

▶️ Running the Chatbot Locally

Activate your virtual environment

.venv\Scripts\activate       # Windows
# or
source .venv/bin/activate    # macOS/Linux


Run the app

streamlit run app.py


Open the local URL shown in the terminal:
👉 http://localhost:8501

Enter Urdu text such as:

السلام علیکم! کیا حال ہے؟
آج موسم کیسا ہے؟
تمہیں کس نے بنایا؟

💬 Sample Interaction

User:

السلام علیکم! کیا حال ہے؟


Chatbot:

وعلیکم السلام! میں ٹھیک ہوں، آپ کیسے ہیں؟


(Responses may vary depending on model training.)

🧪 Training Details (from nlpass2.ipynb)

Dataset: 20,000 Urdu sentence pairs (normalized + cleaned)

Tokenizer: SentencePiece (vocab_size=16000)

Split: Train 80%, Val 10%, Test 10%

Epochs: 10 (extendable to 25+)

Batch Size: 32

Learning Rate: 1e-4

Loss: Cross Entropy

Metrics: BLEU, ROUGE-L, chrF, Perplexity

The model captures Urdu linguistic structure successfully but requires longer training for fluent and contextually accurate replies.

🌐 Streamlit UI Features

Right-to-Left Urdu input box

Display of conversation history

Sidebar configuration:

Checkpoint path (urdu_chatbot.pt)

Tokenizer path (urdu_tokenizer.model)

Decoding mode (Greedy / Beam Search)

Adjustable beam size & max token length

Real-time generation feedback with spinners

Clean conversational layout

🧭 How to Deploy on Streamlit Cloud

Push all project files to GitHub
👉 https://github.com/Usman-Ifty/Urdu-Chatbot-Transformer

Go to https://share.streamlit.io

Connect your GitHub account

Select repository: Usman-Ifty/Urdu-Chatbot-Transformer

Entry file: app.py

Click Deploy

You’ll receive a live link such as:
https://usman-ifty-urdu-chatbot-transformer.streamlit.app

⚙️ Troubleshooting
Issue	Cause	Fix
Random Urdu words	Undertrained model (10 epochs)	Retrain for ≥ 25 epochs
Streamlit shows “state_dict mismatch”	FeedForward layer renamed	Use w1/w2 instead of linear1/linear2
Tokenizer loads but output gibberish	Wrong .model file	Use the exact same tokenizer from training
Long unreadable output	Missing EOS ID	Ensure BOS=2, EOS=3
📈 Future Work

Fine-tune model for 25+ epochs

Add more Urdu conversational data

Visualize attention heatmaps

Implement top-k / nucleus sampling decoding

Improve contextual memory across turns

Deploy fully on Streamlit Cloud with persistent chat history

🧑‍💻 Author

Muhammad Usman Awan
FAST-NUCES, Chiniot-Faisalabad Campus
BSCS – Semester 7
GitHub: Usman-Ifty

📜 License

MIT License © 2025 Muhammad Usman Awan

⭐ GitHub Repository

🔗 https://github.com/Usman-Ifty/Urdu-Chatbot-Transformer