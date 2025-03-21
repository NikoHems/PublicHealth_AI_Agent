import streamlit as st
from main import filtered_df, scenario_percentages, run_conversation, conversation_history
import math

# -------------------------------
# Streamlit UI Setup
# -------------------------------

st.title("Epidemiology Data Analysis Tool")
st.sidebar.header("Settings")

# Allow the user to select a scenario percentage.
selected_scenario = st.sidebar.selectbox(
    "Select Scenario Percentage:",
    [int(pct * 100) for pct in scenario_percentages]
)

# Calculate the number of rows based on the selected scenario.
selected_pct = selected_scenario / 100
num_rows = math.ceil(len(filtered_df) * selected_pct)
snippet_df = filtered_df.head(num_rows)
data_snippet = snippet_df.to_csv(index=False)

# Display a preview of the data.
st.subheader(f"Data Preview ({selected_scenario}% of data)")
st.dataframe(snippet_df.head(5))

# Display the conversation transcript.
st.subheader("Conversation Transcript")

# Run the conversation and generate responses.
conversation_history, mediator_response = run_conversation(data_snippet, conversation_history)

# Show the mediator's final response.
st.subheader("Mediator's Final Response")
st.markdown(f"**Mediator:** {mediator_response}")

# Show the full conversation transcript.
st.subheader("Full Conversation Transcript")
st.markdown(conversation_history)
