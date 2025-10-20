🗣️ Urdu Chatbot — Transformer with Multi-Head Attention

A Transformer-based Urdu Conversational Chatbot built completely from scratch using PyTorch, trained on an Urdu dialogue dataset and deployed with a Streamlit web interface for real-time interaction.

This project demonstrates an end-to-end NLP pipeline — including data preprocessing, tokenization, model training, evaluation, and deployment — customized for Urdu.

🧠 Project Overview

The aim of this project is to design and implement a Transformer Encoder–Decoder architecture that can understand and generate Urdu text without relying on pre-trained models.

It integrates:

Multi-Head Attention for contextual understanding

Positional Encoding for word order retention

SentencePiece Tokenization for subword-level Urdu text

Streamlit UI for real-time Urdu chatbot interaction

🚀 Key Features

Transformer Encoder–Decoder model from scratch

Multi-Head Attention mechanism

SentencePiece-based Urdu tokenizer

Streamlit app for real-time interaction

Beam Search & Greedy decoding support

Right-to-Left Urdu rendering

Modular PyTorch implementation

🏗️ Architecture Details
Component	Description
Embedding Dimension	128
Encoder Layers	2
Decoder Layers	2
Attention Heads	4
Feed-Forward Dimension	512
Dropout	0.2
BOS / EOS IDs	2 / 3
Optimizer	Adam (lr = 1e-4)
Loss Function	Cross-Entropy Loss
📂 Repository Structure
Urdu-Chatbot-Transformer/
│
├── app.py                   # Streamlit chatbot interface
├── urdu_chatbot.pt          # Trained Transformer model
├── urdu_tokenizer.model     # SentencePiece tokenizer
├── normalized_sentences.csv # Urdu dataset
├── nlpass2.ipynb            # Training notebook
└── README.md                # Project documentation

🧰 Requirements

Install all dependencies:

pip install torch streamlit sentencepiece sacrebleu rouge-score

▶️ How to Run

Activate your virtual environment:

.venv\Scripts\activate      # Windows
# or
source .venv/bin/activate   # macOS/Linux


Run the Streamlit app:

streamlit run app.py


Open the local URL (e.g. http://localhost:8501) in your browser.

💬 Example Urdu Prompts
السلام علیکم! کیا حال ہے؟
آج موسم کیسا ہے؟
تمہیں کس نے بنایا؟
کیا تم اردو سمجھ سکتے ہو؟

🧪 Training Configuration
Parameter	Value
Dataset	Urdu Conversational Dataset (20 K pairs)
Tokenizer	SentencePiece (vocab = 16000)
Split	Train 80%, Val 10%, Test 10%
Epochs	10
Batch Size	32
Learning Rate	1e-4
Loss Function	Cross-Entropy
Metrics	BLEU, ROUGE-L, chrF, Perplexity
🧩 Example Output

User:

السلام علیکم! کیا حال ہے؟


Chatbot:

وعلیکم السلام! میں ٹھیک ہوں، آپ کیسے ہیں؟


(Responses may vary depending on training; current checkpoint produces syntactically valid Urdu but still needs semantic fine-tuning.)

🌐 Deployment on Streamlit Cloud

Push all files to GitHub:
👉 https://github.com/Usman-Ifty/Urdu-Chatbot-Transformer

Visit https://share.streamlit.io

Connect your GitHub account

Select this repo → choose app.py → click Deploy

Your app will appear at a public URL like:
https://usman-ifty-urdu-chatbot-transformer.streamlit.app

⚙️ Troubleshooting
Issue	Cause	Fix
Random Urdu words	Under-trained model	Retrain for ≥ 25 epochs
“state_dict mismatch”	Layer name changes	Align linear1/linear2 with w1/w2
Garbled text	Wrong tokenizer	Use exact .model used in training
Endless output	Wrong EOS ID	Ensure BOS = 2 and EOS = 3
📈 Future Work

Continue training for 25+ epochs

Add larger and cleaner Urdu datasets

Visualize attention weights

Introduce temperature / top-k sampling

Add conversation memory

Deploy fully on Streamlit Cloud

👤 Author

Muhammad Usman Awan
FAST-NUCES | Chiniot-Faisalabad Campus
BSCS – Semester 7
GitHub: Usman-Ifty

📜 License

MIT License © 2025 Muhammad Usman Awan

⭐ GitHub Repository

🔗 https://github.com/Usman-Ifty/Urdu-Chatbot-Transformer
