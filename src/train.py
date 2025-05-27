import pandas as pd
import numpy as np
from data_fetcher import fetch_stock_data, save_stock_data
from llm_feature_generator import generate_llm_features
from rl_agent import QLearningAgent

def calculate_reward(action: int, current_price: float, next_price: float) -> float:
    """Calculate reward based on action and price change."""
    if action == 0:  # Buy
        return (next_price - current_price) / current_price * 100
    elif action == 1:  # Sell
        return (current_price - next_price) / current_price * 100
    else:  # Hold
        return 0

def main():
    ticker = "AAPL"
    start_date = "2023-01-01"
    end_date = "2025-01-01"
    df = fetch_stock_data(ticker, start_date, end_date)
    df = generate_llm_features(df)
    save_stock_data(df, "data/stock_data.csv")

    state_size = 100
    action_size = 3
    agent = QLearningAgent(state_size, action_size)

    episodes = 1000
    total_rewards = []
    for episode in range(episodes):
        episode_reward = 0
        for i in range(len(df) - 1):
            state = np.array([df['Returns'].iloc[i], df['Sentiment'].iloc[i]])
            action = agent.choose_action(state)
            reward = calculate_reward(action, df['Close'].iloc[i], df['Close'].iloc[i + 1])
            next_state = np.array([df['Returns'].iloc[i + 1], df['Sentiment'].iloc[i + 1]])
            agent.learn(state, action, reward, next_state)
            episode_reward += reward
        total_rewards.append(episode_reward)
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {episode_reward:.2f}, Exploration Rate: {agent.exploration_rate:.3f}")

    print(f"Average Reward: {np.mean(total_rewards):.2f}")

if __name__ == "__main__":
    main()
