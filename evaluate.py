import os
import re
import pandas as pd
from collections import defaultdict

# === CONFIG ===
REASONING_KEYWORDS_LEVEL_2 = [
    "data shows", "data suggests", "trend indicates", "trend", "based on the numbers", "increase in cases",
    "rise in deaths", "decline in cases", "surge", "spike", "mortality", "hospitalizations are rising",
    "threshold has been crossed", "variant of concern", "epidemiological", "r-value", "transmission rate",
    "exponential growth", "lag in reporting", "flattening the curve", "model predicts"
]

REASONING_KEYWORDS_LEVEL_1 = [
    "we should be careful", "i agree", "we must act fast", "i support this", "it's urgent", "let's proceed",
    "that makes sense", "we should take action", "we need to be prepared", "public health is important",
    "we must protect people", "that’s concerning", "it’s important to consider"
]

SILENCE_PHRASE = "i remain silent this round"

DISAGREEMENT_INDICATORS = [
    "i disagree", "that's incorrect", "that's not true", "you're wrong", "i don’t think that’s the case", "that's not quite right", "no, because",
    "however", "but", "although", "yet", "still", "whereas", "on the contrary", "conversely",
    "actually", "in fact", "let me clarify", "allow me to correct", "the data says otherwise", "to be precise", "that’s a misconception",
    "i’m not sure that’s accurate", "are you sure", "i find that hard to believe", "it’s questionable whether", "there’s no evidence for that", "that assumption might be flawed", "one could argue the opposite",
    "instead, we should", "another way to look at it is", "i would propose", "an alternative might be", "let’s consider a different approach", "we could also"
]

# === FUNCTIONS ===
def clean_and_split_transcript(content):
    content = content.split("**Final Evaluation:**")[0]
    content = content.replace("**", "")
    content = re.sub(r"\n{2,}", "\n", content)

    agent_lines = []
    for line in content.splitlines():
        if ":" in line:
            parts = line.split(":", 1)
            agent = parts[0].strip()
            msg = parts[1].strip()
            if agent and msg:
                agent_lines.append((agent, msg))
    return agent_lines

def score_reasoning(msg):
    msg = msg.lower()
    if SILENCE_PHRASE in msg:
        return 0
    if any(kw in msg for kw in REASONING_KEYWORDS_LEVEL_2):
        return 2
    if any(kw in msg for kw in REASONING_KEYWORDS_LEVEL_1):
        return 1
    return 1

def has_disagreement(msg):
    msg = msg.lower()
    return int(any(kw in msg for kw in DISAGREEMENT_INDICATORS))  # max 1 per message

def evaluate_transcript(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    entries = clean_and_split_transcript(content)

    reasoning_scores = []
    disagreement_count = 0

    for _, msg in entries:
        reasoning_scores.append(score_reasoning(msg))
        disagreement_count += min(1, has_disagreement(msg))  # enforce max 1 per message

    avg_reasoning = round(sum(reasoning_scores) / len(reasoning_scores), 2) if reasoning_scores else 0.0

    return {
        "Filename": os.path.basename(path),
        "Reasoning Depth (Avg)": avg_reasoning,
        "Disagreements": disagreement_count
    }

def evaluate_folder(folder_path):
    results = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".txt"):
            fpath = os.path.join(folder_path, fname)
            results.append(evaluate_transcript(fpath))
    return pd.DataFrame(results)

if __name__ == "__main__":
    folder = "./manipulated_data"
    df = evaluate_folder(folder)
    df.to_csv("final_disagreement_capped_results.csv", index=False)
    print("Saved to final_disagreement_capped_results.csv")
