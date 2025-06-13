# Simple reward model for demonstration: prefers longer, more detailed completions

def reward_score(completion):
    """Assigns a reward score based on length and some keywords."""
    score = len(completion.split())
    if any(word in completion.lower() for word in ["helpful", "honest", "harmless"]):
        score += 10
    return score

if __name__ == "__main__":
    # Demo
    a = "This is a helpful, honest answer."
    b = "Short reply."
    print(f"A: {reward_score(a)} | B: {reward_score(b)}")
