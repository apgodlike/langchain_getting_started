from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig
from langchain_openai import OpenAIEmbeddings

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool
from langchain.agents.middleware import dynamic_prompt, ModelRequest


load_dotenv()



# pdf_path = "workout.pdf"
# loader = PyPDFLoader(pdf_path)

# documents = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,  # chunk size (characters)
#     chunk_overlap=200,  # chunk overlap (characters)
#     add_start_index=True,  # track index in original document
# )

# all_splits = text_splitter.split_documents(documents)
# print(f"Split blog post into {len(all_splits)} sub-documents.")



embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Chroma(
    # documents=all_splits,
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db_persist",  # Where to save data locally, remove if not necessary
)


# document_ids = vector_store.add_documents(documents=all_splits)

# vector_store.persist()


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


tools = [retrieve_context]

prompt = (
    "You have access to a tool that retrieves context from a vector db. If you cant find answer here, do not answer the question.\n Only use the information provided by the tool to answer the question.\n\n"
)




# query = (
#     "What is the standard method for Task Decomposition?\n\n"
#     "Once you get the answer, look up common extensions of that method."
# )

# for event in agent.stream(
#     {"messages": [{"role": "user", "content": query}]},
#     stream_mode="values",
# ):
#     event["messages"][-1].pretty_print()


@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context into state messages."""
    last_query = request.state["messages"][-1].text
    # print(f"Last query: {last_query}")
    # print("=======================================================")
    retrieved_docs = vector_store.similarity_search(last_query)
    # print(f"Retrieved {len(retrieved_docs)} documents from vector store.")
    # print("=======================================================")

    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    system_message = (
        "You have access to a tool that retrieves context from a vector db. If you cant find answer here, do not answer the question.\n Only use the information provided by the tool to answer the question.\n\n"
        f"\n\n{docs_content}"
    )

    return system_message

agent = create_agent(
    model="gpt-5-nano",
    # checkpointer=InMemorySaver(),
    tools=tools,
    system_prompt = prompt,
    middleware=[prompt_with_context]
    )

def runner():
    query = input("Enter your query: ")
    for step in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()

    runner()
runner()





# comment


















# config: RunnableConfig = {"configurable": {"thread_id": "1"}}
def chat():
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting chat.")
        return


    response = agent.invoke({"messages":user_input}, config)
    print("Response:")
    print(response["messages"][-1].pretty_print())

    chat()
# chat()




# retriver = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

