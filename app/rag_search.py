"""
Module for Retrieval-Augmented Generation (RAG).

This module handles the interaction with the Large Language Model (LLM) to generate
natural language answers based on the video context retrieved from the database.
"""

import os
import sys
from typing import List, Optional

from openai import OpenAI

# Ensure project root is in sys.path for standalone execution
try:
    from app.config import Config
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from app.config import Config

# Initialize OpenAI Client
client = OpenAI(api_key=Config.OPENAI_API_KEY)


def generate_answer(user_query: str, relevant_chunks: List[str]) -> str:
    """
    Generates a concise answer to the user's question using retrieved video context.

    Args:
        user_query (str): The question asked by the user.
        relevant_chunks (List[str]): A list of text strings retrieved from the
                                     vector database (the context).

    Returns:
        str: The generated answer from the LLM.
    """
    if not relevant_chunks:
        return "I could not find enough information in the videos to answer that."

    # 1. Context Assembly
    # Combine the retrieved text chunks into a single formatted block.
    # We use a bulleted list format to help the model distinguish between separate segments.
    context_text = ""
    for chunk in relevant_chunks:
        context_text += f"- {chunk}\n"

    # 2. Prompt Construction
    # The System Prompt defines the persona and rules (be factual, concise).
    system_prompt = (
        "You are a helpful news assistant. You will be given a user question and "
        "several context snippets from video transcripts and visual descriptions.\n"
        "Your job is to answer the question based ONLY on the provided context.\n"
        "If the context does not contain the answer, say 'I do not have that information.'\n"
        "Keep the answer concise (2-3 sentences)."
    )

    # The User Message contains the dynamic data for this specific request.
    user_message = f"""
    Context from videos:
    {context_text}

    Question: {user_query}
    """

    # 3. LLM Generation
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Using GPT-4o for high-quality reasoning
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3  # Low temperature reduces hallucinations/creativity
        )
        
        answer = response.choices[0].message.content
        return answer if answer else "Error generating response."

    except Exception as e:
        print(f"Error generating RAG answer: {e}")
        return "An error occurred while generating the answer."


if __name__ == "__main__":
    # Test block to verify RAG logic independent of the frontend
    test_query = "What is the update on the peace talks?"
    
    # Mock data simulating what ChromaDB would return
    test_context = [
        "[Visual Scene]: Politicians shaking hands. [Audio Transcript]: The delegation met in Yevlakh today.",
        "[Visual Scene]: A document is signed. [Audio Transcript]: They discussed reintegration plans."
    ]
    
    print(f"Testing RAG generation for query: '{test_query}'")
    result = generate_answer(test_query, test_context)
    print(f"AI Answer: {result}")