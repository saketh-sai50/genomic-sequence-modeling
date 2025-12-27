from genai.prompts import EXPLANATION_PROMPT

def generate_explanation(sequence, prediction):
    """
    Generates a human-readable explanation for a model prediction
    using an LLM (API call simulated here).
    """

    prompt = EXPLANATION_PROMPT.format(
        sequence=sequence,
        prediction="Splice Site" if prediction == 1 else "Non-Splice Site"
    )

    # ---- LLM API CALL (example placeholder) ----
    # response = llm_api.generate(prompt)
    # explanation = response.text

    # Placeholder explanation (safe for now)
    explanation = (
        "The model identified recurring nucleotide patterns and contextual "
        "dependencies in the sequence that are commonly associated with "
        "splice-related regions."
        if prediction == 1
        else
        "The sequence lacks consistent nucleotide patterns that the model "
        "learned to associate with splice-related regions."
    )

    return explanation
