import os
import streamlit as st
import math
import pandas as pd
from dotenv import load_dotenv
import datetime

# Load environment variables from a .env file if present.
load_dotenv()

# Ensure the API key is set via environment variable.
if "OPENAI_API_KEY" not in os.environ:
    st.error("OPENAI_API_KEY not found. Please set it in your environment or in a .env file.")
    st.stop()

from main import filtered_df, scenario_percentages, create_agents, run_conversation

st.title("Epidemiology Data Analysis Tool")
st.sidebar.header("Settings")

# Toggle between normal and manipulated datasets.
dataset_option = st.sidebar.radio("Select Dataset", ("Normal", "Incomplete"))

selected_scenario = st.sidebar.selectbox(
    "Select Scenario Percentage:",
    [int(pct * 100) for pct in scenario_percentages]
)
selected_pct = selected_scenario / 100

# Load the dataset based on the selected option.
if dataset_option == "Normal":
    st.sidebar.info("Using the normal dataset.")
    base_df = filtered_df
else:
    st.sidebar.info("Using the manipulated dataset.")
    base_df = pd.read_csv("germany_epidemiology_data_noisy_incomplete.csv")

# Slice data according to the chosen scenario percentage.
num_rows = math.ceil(len(base_df) * selected_pct)
snippet_df = base_df.head(num_rows)
data_snippet = snippet_df.to_csv(index=False)

st.subheader(f"Data Preview ({selected_scenario}% of data)")
st.dataframe(snippet_df.head(5))

st.subheader("Conversation Transcript")

# Create conversation agents.
agents, evaluator = create_agents()
conversation_history, evaluation = run_conversation(
    data_snippet, "Initial conversation transcript:\n\n", agents, evaluator
)

# Generate a timestamp for the filename.
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"conversation_transcript_{timestamp}.txt"

# Save the conversation transcript with a unique filename.
with open(filename, "w") as file:
    file.write(conversation_history)

st.subheader("Final Evaluation")
st.markdown(f"**Final Evaluation:** {evaluation}")

st.subheader("Full Conversation Transcript")
st.markdown(conversation_history)
