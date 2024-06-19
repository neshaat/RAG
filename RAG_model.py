import sys
import os
import requests
from requests.exceptions import ReadTimeout, ConnectionError
from datasets import load_dataset  # Import the datasets library

import dspy

class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)

class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

def configure_dspy():
    ollama = dspy.OllamaLocal(model='llama3')
    colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
    dspy.settings.configure(lm=ollama, rm=colbertv2_wiki17_abstracts)

def check_ollama_server():
    try:
        response = requests.get('http://localhost:11434/api/generate', timeout=10)
        if response.status_code == 200:
          print("OllamaLocal server is running.")
        else:
          print("OllamaLocal server responded with an error.")
    except (ReadTimeout, ConnectionError):
        print("OllamaLocal server is not accessible. Please ensure it is running on port 11434.")
        raise

def ask_question(rag_model, question):
    try:
        prediction = rag_model.forward(question)
        print(f"Question: {question}")
        print(f"Predicted Answer: {prediction.answer}")
        print(f"Context: {prediction.context}")
        print('-' * 60)
    except (ReadTimeout, ConnectionError) as e:
        print(f"Failed to generate prediction for question '{question}': {e}")

def main():
    check_ollama_server()
    configure_dspy()

    # Load SQuAD dataset
    squad = load_dataset('squad')

    # Prepare the data (example from SQuAD)
    train_questions = squad['train']['question']
    train_answers = squad['train']['answers']

    # Define a new RAG model
    rag_model = RAG(num_passages=3)

    # Ask questions from the SQuAD dataset (for demonstration, only first 5 questions)
    for i in range(5):
        question = train_questions[i]
        ask_question(rag_model, question)

if __name__ == "__main__":
    main()



