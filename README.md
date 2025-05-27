# Quantitative-Stock-Analysis-with-Reinforcement-Learning-and-LLM
This project implements a reinforcement learning (RL) approach for quantitative stock analysis, where Markov Decision Processes (MDPs) are enhanced by features generated from an open-source LLM. The agent uses Q-learning to make buy/sell/hold decisions based on historical stock data and LLM-generated sentiment scores from synthetic news snippets.


Project Structure





src/: Source code for data fetching, LLM feature generation, RL agent, and training.



data/: Stores historical stock data (e.g., stock_data.csv).



requirements.txt: Lists Python dependencies.



.gitignore: Ignores unnecessary files (e.g., __pycache__, .env).

Setup





Clone the repository:

git clone https://github.com/yourusername/quant_stock_analysis_rl.git
cd quant_stock_analysis_rl



Install dependencies:

pip install -r requirements.txt



(Optional) For LLaMA: Set up a Hugging Face account and obtain an API token or download LLaMA weights (requires approval from Meta AI). Update llm_feature_generator.py with your token or model path.



Run the training script:

python src/train.py

Dependencies





Python 3.8+



Libraries: yfinance, pandas, numpy, transformers, torch



Optional: LLaMA model weights or Hugging Face API token for advanced LLM usage

Usage





The script fetches historical stock data for AAPL (configurable in train.py).



LLM features are generated in llm_feature_generator.py using DistilBERT by default. To use LLaMA, modify the script to load the model via Hugging Face's transformers or a local setup.



The RL agent (rl_agent.py) uses Q-learning to optimize trading decisions.



Results are printed to the console, including total reward and actions taken.

Using LLaMA





This project uses DistilBERT for lightweight sentiment analysis. To use LLaMA:





Install transformers and torch with GPU support if available.



Obtain LLaMA weights from Meta AI or use a Hugging Face model like meta-llama/Llama-3.1-8B.



Update llm_feature_generator.py to load LLaMA and adjust the sentiment analysis logic.



Ensure sufficient GPU memory (e.g., 16GB+ for LLaMA 3.1-8B).



Example modification for LLaMA:

from transformers import LlamaForSequenceClassification, LlamaTokenizer
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
model = LlamaForSequenceClassification.from_pretrained("meta-llama/Llama-3.1-8B")

Notes





The LLM processes synthetic news snippets to generate sentiment scores. In production, integrate real-time data from X or news APIs.



The MDP state space includes stock prices, technical indicators, and LLM-generated sentiment.



Adjust hyperparameters in train.py for better performance.



Due to LLaMA's resource requirements, DistilBERT is used by default for accessibility.
