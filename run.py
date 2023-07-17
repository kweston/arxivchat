# %%
with open('example.md', 'r') as f:
    md_file = f.read()


# Let's create groups based on the section headers in our page
from langchain.text_splitter import MarkdownHeaderTextSplitter
headers_to_split_on = [
    ("###", "Section"),
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splits = markdown_splitter.split_text(md_file)

# convert section values to lowercase so that they match the queries
for doc in md_header_splits:
    if 'Section' in doc.metadata:
        doc.metadata['Section'] = doc.metadata['Section'].lower()

# Define our text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
chunk_size = 500
chunk_overlap = 0
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
all_splits = text_splitter.split_documents(md_header_splits)


# Build vectorstore and keep the metadata
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
vectorstore = Chroma.from_documents(documents=all_splits,
                                    embedding=OpenAIEmbeddings())

# Create retriever 
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

# Define our metadata
metadata_field_info = [
    AttributeInfo(
        name="Section",
        description="Part of the document that the text comes from",
        type="string or list[string]",
    ),
]
document_content_description = "Major sections of the document"

# Define self query retriver
llm = OpenAI(temperature=0)
retriever = SelfQueryRetriever.from_llm(llm, vectorstore, document_content_description, metadata_field_info, verbose=True)
out = retriever.get_relevant_documents("Summarize the motivation section of the document")
print(out)
pass


# %%
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(temperature=0, verbose=True)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, verbose=True, return_source_documents=True)
#out = qa_chain("Summarize the motivation section of this document")
out = qa_chain("Summarize this document")
print(out)
pass




# %%
