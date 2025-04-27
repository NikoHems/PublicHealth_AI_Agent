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
# Helper Function
# -------------------------------

def extract_clean_response(raw_response) -> str:
    """
    Extract and clean the response content from the raw LLM output.
    
    Args:
        raw_response: The raw response object returned by the LLM chain.
    
    Returns:
        str: A stripped string extracted from the raw response.
    """
    if hasattr(raw_response, 'content'):
        return raw_response.content.strip()
    return str(raw_response).strip()

# -------------------------------
# Agent Classes Definitions
# -------------------------------

class ConversationAgent:
    """
    Represents a conversation agent that generates responses based on role-specific instructions.
    """
    def __init__(self, name: str, role_instruction: str, model_name: str = "gpt-4o-mini", temperature: float = 0.7):
        """
        Initializes a ConversationAgent.
    
        Args:
            name (str): The agent's name.
            role_instruction (str): Role instructions including directives to engage with colleagues.
            model_name (str, optional): The LLM model name. Defaults to "gpt-4o-mini".
            temperature (float, optional): Sampling temperature for the LLM. Defaults to 0.7.
        """
        self.name = name
        full_instruction = role_instruction + " Additionally, directly reference and respond to your colleagues' previous contributions."
        prompt = PromptTemplate(
            input_variables=["data", "previous_responses"],
            template=(
                "Data:\n{data}\n\nPrevious Responses:\n{previous_responses}\n\n"
                "Role instructions: " + full_instruction + "\n\nResponse:"
            )
        )
        self.chain = prompt | ChatOpenAI(model_name=model_name, temperature=temperature)
    
    def get_response(self, data: str, previous_responses: str) -> str:
        """
        Generates a response using the underlying LLM chain.
    
        Args:
            data (str): The data snippet.
            previous_responses (str): The conversation transcript so far.
    
        Returns:
            str: The agent's clean response.
        """
        raw_response = self.chain.invoke({"data": data, "previous_responses": previous_responses})
        return extract_clean_response(raw_response)

class EvaluatorAgent:
    """
    Represents an evaluation agent that reviews the overall conversation transcript along with the underlying data.
    It produces a final evaluation summary that identifies which agents are most consistent with the data,
    and then decides whether a consensus was reached.
    """
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.7):
        """
        Initializes the EvaluatorAgent.
    
        Args:
            model_name (str, optional): The LLM model name. Defaults to "gpt-4o-mini".
            temperature (float, optional): Sampling temperature for the LLM. Defaults to 0.7.
        """
        prompt = PromptTemplate(
            input_variables=["conversation", "data"],
            template=(
                "Full Conversation Transcript:\n{conversation}\n\n"
                "Underlying Data:\n{data}\n\n"
                "Evaluator Instructions: Review the conversation above and compare the agents' contributions with the underlying epidemiology data. "
                "Evaluate which agents' arguments are well-supported and consistent with the data. Provide a final analytical summary that does not include numerical scores "
                "but clearly states which agents form the best team (i.e., which are most consistent with the data). "
                "Finally, decide whether the agents reached consensus. Conclude your response with either 'Consensus reached: Yes' or 'Consensus reached: No'.\n\nResponse:"
            )
        )
        self.chain = prompt | ChatOpenAI(model_name=model_name, temperature=temperature)
    
    def get_evaluation(self, conversation: str, data: str) -> str:
        """
        Generates a final evaluation summary based on the conversation transcript and the underlying data.
    
        Args:
            conversation (str): The complete conversation transcript.
            data (str): The underlying data snippet as CSV text.
    
        Returns:
            str: The evaluator's summary.
        """
        raw_response = self.chain.invoke({"conversation": conversation, "data": data})
        return extract_clean_response(raw_response)

# -------------------------------
# Agent Creation Function
# -------------------------------

def create_agents():
    """
    Dynamically creates conversation agents and the evaluator agent after the API key is set.
    
    Returns:
        tuple: (list of ConversationAgent instances, EvaluatorAgent instance)
    """
    agents = [
        ConversationAgent(
            name="Epidemiologist",
            role_instruction=(
                "You are an experienced epidemiologist. Analyze the data and respond directly to your colleagues’ arguments. "
                "Vary your response length between 10 and 100 words. If you have nothing new to add, state 'I remain silent this round', be sure to only use this rarely and try to contribute if you can. "
                "Challenge points you disagree with using evidence when possible."
            )
        ),
        ConversationAgent(
            name="Public Health Worker",
            role_instruction=(
                "You are a pragmatic public health worker. Engage with your colleagues by offering counterpoints and defending your proposals. "
                "Your response should vary between 10 and 100 words. If you have nothing new to add, state 'I remain silent this round', be sure to only use this rarely and try to contribute if you can."
            )
        ),
        ConversationAgent(
            name="Medical Doctor",
            role_instruction=(
                "You are a medical doctor focusing on clinical risks. Respond directly to your peers with evidence-based counterpoints. "
                "Keep your response between 10 and 100 words. If you have nothing new to add, state 'I remain silent this round', be sure to only use this rarely and try to contribute if you can."
            )
        ),
        ConversationAgent(
            name="Public Health Worker (Cautious)",
            role_instruction=(
                "You are a public health worker who is extremely cautious about imposing restrictions. Argue that measures should only be taken after a clear threshold is crossed. "
                "Respond directly to others’ points using 10 to 100 words. If you have nothing new to add, state 'I remain silent this round', be sure to only use this rarely and try to contribute if you can."
            )
        ),
        ConversationAgent(
            name="Head of the RKI",
            role_instruction=(
                "You are the Head of the RKI, the highest authority in this discussion. Your opinion is final. "
                "Respond decisively to the debate using 10 to 100 words and address your colleagues' points. "
                "If you have nothing further to add, you may still provide a brief confirmation."
            )
        )
    ]
    evaluator = EvaluatorAgent()
    return agents, evaluator

# -------------------------------
# Conversation Runner
# -------------------------------

def run_conversation(data_snippet: str, conversation_history: str, agents, evaluator_agent) -> (str, str):
    """
    Conducts a multi-round conversation among agents and produces a final evaluation summary.
    
    The conversation proceeds in rounds (minimum 2, maximum 6) where each agent contributes a response.
    Agents are expected to directly engage with each other's contributions.
    In the first round, agents are not allowed to be silent; if an agent returns "I remain silent this round",
    the system re-prompts that agent with a reminder.
    After the conversation rounds finish, the evaluator agent reviews the transcript along with the underlying data 
    and produces a final evaluation including whether a consensus was reached.
    
    Args:
        data_snippet (str): CSV string containing the underlying data.
        conversation_history (str): The conversation transcript so far.
        agents (list): List of ConversationAgent instances.
        evaluator_agent (EvaluatorAgent): An instance of EvaluatorAgent.
    
    Returns:
        tuple:
            - conversation_history (str): The complete conversation transcript including all rounds.
            - evaluation (str): The final evaluation summary produced by the evaluator agent.
    """
    min_rounds = 2
    max_rounds = 6
    round_num = 1

    while round_num <= max_rounds:
        round_transcript = f"--- Round {round_num} ---\n"
        new_contribution = False
        
        for agent in agents:
            response = agent.get_response(data_snippet, conversation_history)
            
            # Force a non-silent response in round 1.
            if round_num == 1 and response.lower() == "i remain silent this round":
                modified_history = conversation_history + "\nNote: In Round 1, you must provide an active contribution. Silence is not allowed. Elaborate your thought."
                response_retry = agent.get_response(data_snippet, modified_history)
                if response_retry.lower() != "i remain silent this round":
                    response = response_retry
            
            round_transcript += f"**{agent.name}:** {response}\n\n"
            if response.lower() != "i remain silent this round":
                new_contribution = True
        
        conversation_history += round_transcript + "\n"
        if round_num >= min_rounds and not new_contribution:
            break
        round_num += 1

    # Get the final evaluation, including consensus decision.
    evaluation = evaluator_agent.get_evaluation(conversation_history, data_snippet)
    conversation_history += f"**Final Evaluation:** {evaluation}\n\n"
    
    return conversation_history, evaluation


if __name__ == "__main__":
    # For command-line testing.
    data_snippet = filtered_df.head(100).to_csv(index=False)
    agents, evaluator = create_agents()
    final_transcript, final_evaluation = run_conversation(data_snippet, "Initial conversation transcript:\n\n", agents, evaluator)
    print(final_transcript)
