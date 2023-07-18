from arxiv_chat.arxivpdf import ArxivPDF
import argparse


def main(fname, live_input=True):

    pdf = ArxivPDF(fname)
    docs = pdf.split_text(split_sections=True)
    # Define our text splitter
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    all_splits = text_splitter.split_documents(docs)


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
    # out = retriever.get_relevant_documents("Summarize the introduction section of the document")
    # print(out)


    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI
    from langchain.memory import ConversationBufferMemory

    llm = ChatOpenAI(temperature=0, verbose=True)
    qa_chain = RetrievalQA.from_chain_type(
        llm, retriever=retriever,
        verbose=True,
        return_source_documents=False,
        memory=ConversationBufferMemory()
    )
    #out = qa_chain("Summarize the motivation section of this document")
    if live_input:
        while True:
            question = input("Enter a question (q to quit): ")
            if question.strip() == "q":
                break
            answer = qa_chain(question)
            print(f"query: {answer['query']}")
            print(f"result: {answer['result']}")
    else:
        out = qa_chain("Give a concise summary of this document")
        print(out)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run doc QA on the given Arxiv PDF")
    parser.add_argument("fname", type=str, help="Path to the PDF file")
    args = parser.parse_args(["2302.00923v4_clean.pdf"])
    main(args.fname)
