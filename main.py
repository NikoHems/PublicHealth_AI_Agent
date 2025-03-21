import pandas as pd
import math
import random
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

# -------------------------------
# Data Loading and Filtering
# -------------------------------

# Load the epidemiology data and filter for Germany.
data_df = pd.read_csv("Epidemiology Data.csv")
germany_df = data_df[data_df['location_key'] == 'DE']

# Select relevant columns.
filtered_df = germany_df[['date', 'new_confirmed', 'new_deceased', 
                            'cumulative_confirmed', 'cumulative_deceased']]

# Scenario percentages for different data slices.
scenario_percentages = [0.1, 0.3, 0.6, 1.0]

# -------------------------------
# Agent Chain Definitions
# -------------------------------

def create_agent_chain(role_instruction: str) -> RunnableSequence:
    """
    Create an LLM chain for an agent with a given role instruction.
    
    Args:
        role_instruction (str): Instructions defining the agent's role and style.
        
    Returns:
        RunnableSequence: An LLM chain that uses the provided prompt template.
    """
    prompt = PromptTemplate(
        input_variables=["data", "previous_responses"],
        template=(
            "Data:\n{data}\n\nPrevious Responses:\n{previous_responses}\n\n"
            "Role instructions: " + role_instruction + "\n\nResponse:"
        )
    )
    return prompt | ChatOpenAI(model_name="gpt-4o", temperature=0.7)

# Define agent chains with instructions for direct interaction and varied response lengths.
epidemiologist_agent = create_agent_chain(
    "You are an experienced epidemiologist. Analyze the data and respond directly to your colleagues’ arguments. "
    "Vary your response length between 10 and 100 words. If you have nothing new to add, state 'I remain silent this round'. "
    "Challenge points you disagree with using evidence when possible."
)

ph_worker_agent = create_agent_chain(
    "You are a pragmatic public health worker. Engage with your colleagues by offering counterpoints and defending your proposals. "
    "Your response should vary between 10 and 100 words. If you have nothing new to add, state 'I remain silent this round'."
)

medical_doctor_agent = create_agent_chain(
    "You are a medical doctor focusing on clinical risks. Respond directly to your peers with evidence-based counterpoints. "
    "Keep your response between 10 and 100 words. If you have nothing new to add, state 'I remain silent this round'."
)

ph_worker_cautious_agent = create_agent_chain(
    "You are a public health worker who is extremely cautious about imposing restrictions. Argue that measures should only be taken after a clear threshold is crossed. "
    "Respond directly to others’ points using 10 to 100 words. If you have nothing new to add, state 'I remain silent this round'."
)

head_rki_agent = create_agent_chain(
    "You are the Head of the RKI, the highest authority in this discussion. Your opinion is final. "
    "Respond decisively to the debate using 10 to 100 words and address your colleagues' points. "
    "If you have nothing further to add, you may still provide a brief confirmation."
)

def get_mediator_chain() -> RunnableSequence:
    """
    Get the mediator chain that reviews the full conversation transcript.
    
    The mediator's instructions:
      - Review the conversation.
      - If every agent's final response aligns with the Head of the RKI’s directive, provide a final summary ending with 'CONSENSUS: YES'.
      - Otherwise, summarize the debate without consensus.
      
    Returns:
        RunnableSequence: An LLM chain with the mediator's prompt.
    """
    mediator_template = (
        "Full Conversation Transcript:\n{conversation}\n\n"
        "Mediator Instructions: Review the conversation. If every agent's latest contribution (ignoring 'I remain silent this round') "
        "shows agreement with the Head of the RKI’s directive, provide a final summary ending with 'CONSENSUS: YES'. "
        "Otherwise, summarize the debate and note that no consensus was reached.\n\nResponse:"
    )
    prompt = PromptTemplate(
        input_variables=["conversation"],
        template=mediator_template
    )
    return prompt | ChatOpenAI(model_name="gpt-4o", temperature=0.7)

# Initialize conversation history.
conversation_history = "Initial conversation transcript:\n\n"

# -------------------------------
# Helper Functions
# -------------------------------

def extract_clean_response(raw_response) -> str:
    """
    Extract and clean the response content from the raw LLM output.
    
    Args:
        raw_response: The raw response object from the LLM.
        
    Returns:
        str: The cleaned text from the response.
    """
    if hasattr(raw_response, 'content'):
        return raw_response.content.strip()
    return str(raw_response).strip()

def run_conversation(data_snippet: str, conversation_history: str) -> (str, str):
    """
    Run a multi-round, dynamic conversation among agents.
    
    Agents are prompted each round to decide if they have new arguments to add (if not, they respond with
    "I remain silent this round"). The discussion continues as long as there is new input, with a minimum of 2 rounds
    and a maximum of 6 rounds. If, after round 2, all agents (except optionally the Head of the RKI) remain silent,
    the debate ends early.
    
    At the end, the mediator is invoked to produce a final summary.
    
    Args:
        data_snippet (str): CSV string of the data snippet.
        conversation_history (str): The accumulated conversation transcript.
        
    Returns:
        tuple: A tuple containing the updated conversation transcript and the final mediator response.
    """
    # List of agents as tuples of (name, agent_chain).
    agents = [
        ("Epidemiologist", epidemiologist_agent),
        ("Public Health Worker", ph_worker_agent),
        ("Medical Doctor", medical_doctor_agent),
        ("Public Health Worker (Cautious)", ph_worker_cautious_agent),
        ("Head of the RKI", head_rki_agent)
    ]
    
    min_rounds = 2
    max_rounds = 6
    round_num = 1

    while round_num <= max_rounds:
        round_transcript = f"--- Round {round_num} ---\n"
        new_contribution = False
        
        # Each agent is given the chance to contribute.
        for agent_name, agent_chain in agents:
            response = extract_clean_response(
                agent_chain.invoke({"data": data_snippet, "previous_responses": conversation_history})
            )
            round_transcript += f"**{agent_name}:** {response}\n\n"
            # Consider a contribution "new" if it is not the silence placeholder.
            if response.lower() != "i remain silent this round":
                new_contribution = True
        
        conversation_history += round_transcript + "\n"
        
        # If at least min_rounds are done and no new contributions occurred, end the conversation.
        if round_num >= min_rounds and not new_contribution:
            break
        
        round_num += 1

    # Once the conversation ends (either naturally or by no new input), invoke the mediator.
    mediator_chain = get_mediator_chain()
    mediator_response = extract_clean_response(
        mediator_chain.invoke({"conversation": conversation_history})
    )
    conversation_history += f"**Mediator:** {mediator_response}\n\n"
    
    return conversation_history, mediator_response
