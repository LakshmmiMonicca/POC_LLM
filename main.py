from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

import requests

try:
    res = requests.get("http://localhost:11434")
    if res.status_code == 200:
        print("✅ Ollama server is up!")
    else:
        print("❌ Ollama server not healthy. Status:", res.status_code)
        exit()
except Exception as e:
    print("❌ Ollama not reachable. Error:", e)
    exit()


model = OllamaLLM(
    model="llama3.2",
    base_url="http://localhost:11434"
)


template = """
You are an exeprt in answering questions about a pizza restaurant

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break
    
    documents = retriever.invoke(question)
    reviews = "\n\n".join([doc.page_content for doc in documents])
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)
