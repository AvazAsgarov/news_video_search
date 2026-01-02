"""
Core module for Named Entity Recognition (NER).

This module uses the Spacy library to extract entities such as people,
organizations, and geopolitical locations from text transcripts.
"""

import sys
from typing import Dict, List

import spacy

# Load the pre-trained English model.
nlp = spacy.load("en_core_web_sm")


def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Parses the input text and extracts structured lists of entities.

    Args:
        text (str): The transcript or text content to analyze.

    Returns:
        Dict[str, List[str]]: A dictionary containing lists of unique entities
                              for 'PERSON' (People), 'ORG' (Organizations), and
                              'GPE' (Geopolitical Entities like countries/cities).
    """
    if not text:
        return {"PERSON": [], "ORG": [], "GPE": []}

    doc = nlp(text)

    # Initialize the dictionary with empty lists for the entity types we care about
    entities = {
        "PERSON": [],
        "ORG": [],
        "GPE": []
    }

    # Iterate over entities detected by Spacy
    for ent in doc.ents:
        label = ent.label_
        text_value = ent.text.strip()

        # Filter: We only want PERSON, ORG, and GPE tags
        if label in entities:
            # Deduplication: Avoid adding the same name twice
            if text_value not in entities[label]:
                entities[label].append(text_value)

    return entities


if __name__ == "__main__":
    # Test block to verify NER functionality
    sample_text = (
        "President Ilham Aliyev met with UN officials in Baku regarding COP29. "
        "Lionel Messi scored a goal for Inter Miami in the US."
    )
    
    print("Testing Named Entity Recognition...")
    print(f"Input text: {sample_text}\n")
    
    try:
        results = extract_entities(sample_text)
        print("Extracted Entities:")
        print(f" - People: {results['PERSON']}")
        print(f" - Organizations: {results['ORG']}")
        print(f" - Locations: {results['GPE']}")
    except Exception as e:
        print(f"An error occurred during testing: {e}")