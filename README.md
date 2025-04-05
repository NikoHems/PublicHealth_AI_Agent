### `Public Health AI Agents`

# PublicHealth AI Agent

This project simulates a team of AI agents discussing public health data to collaboratively analyze epidemiological developments and reach consensus or disagreement on interpretation and action. It is designed as an experimental setup to evaluate the ability of AI agents to mimic realistic expert discussions using real and manipulated datasets.

The system includes domain-specific roles (e.g., Epidemiologist, Public Health Worker, Medical Doctor, Head of RKI) powered by OpenAI's GPT models via LangChain. The goal is to explore the potential of AI agents in supporting and simulating decision-making in public health contexts.

## Features

- 📊 Load and preprocess epidemiological data for Germany.
- 🧠 Create multiple domain-specific AI agents using LangChain and OpenAI.
- 💬 Simulate structured, multi-round conversations between agents.
- 🧪 Compare agent behavior on clean vs. noisy/incomplete data.
- ✅ Output full conversation transcripts and evaluations.
- 🌐 Interactive Streamlit interface for selecting data scenarios and viewing results.

## How it Works

1. The user selects a dataset (clean or noisy) and a scenario percentage (10%, 30%, 60%, 100%) from the Streamlit UI.
2. The backend slices the data and feeds it to a group of AI agents, each with a specific role and reasoning behavior.
3. Agents engage in multi-round conversations, debating their interpretation of the data.
4. An Evaluator Agent summarizes the conversation, ranks the agents, and decides if consensus was reached.
5. The conversation transcript and evaluation are saved to a timestamped `.txt` file and displayed in the app.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/NikoHems/PublicHealth_AI_Agent.git
cd PublicHealth_AI_Agent
```

### 2. Set Up Environment

Create a `.env` file in the root directory with your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
streamlit run app.py
```

## Requirements

See [`requirements.txt`](./requirements.txt).

## Output Files

Each run of the simulation saves the full conversation and evaluation summary to a `conversation_transcript_YYYYMMDD_HHMMSS.txt` file for later analysis or validation.

## Example Use Cases

- Testing the robustness of AI-based decision-making in public health.
- Simulating disagreements and consensus among health experts.
- Evaluating how noisy/incomplete data impacts AI-based consensus.

## License

MIT License

## Author

Developed by [Niko Hems](https://github.com/NikoHems) as part of an academic exploration in AI for public health simulation.
