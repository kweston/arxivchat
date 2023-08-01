# Overview 

This app is intended to allow users to chat and ask about research papers published on Arxiv. It uses OpenAI's ChatGPT as a backend but since it was trained in Nov 2022 it doesn't have knowledge of events that occurred after that time. Fine-tuning an LLM such as this tends to be extremely costly in terms of annotation and computing resources and is out of reach for most people. Retrieval augmented generation (RAG) offers a means whereby we can include the specific domain knowledge we are interested in in the context of the query without having to fine-tune the model.

As a proof of concept, I have extracted text from Meta's Llama 2 LLM released on July 18th 2023, and loaded this into a vector store which the model then uses as a source of additional knowledge when answering queries. 

Future work will allow any other research paper to be queried in a similar manner.

# Usage 

To load the baseline chatbot without any specific domain knowledge you can use the --no_tools option
```
python arxivchat.py --live --no_tools
```
Then navigate to http://127.0.0.1:7860/ in a browser. See the example output below.
<img width="831" alt="arxiv_chat_notools" src="https://github.com/kweston/langchain_sandbox/assets/1307463/14526979-ded7-4e6b-9877-373b45d39d5c">


Now compare this to the chat enhanced with specific domain knowledge about the LLama 2 paper using RAG
```
python arxivchat.py --live
```

<img width="754" alt="arxiv_chat" src="https://github.com/kweston/langchain_sandbox/assets/1307463/ef8d44d9-dc41-4e20-bf6e-f3b696294f9a">



