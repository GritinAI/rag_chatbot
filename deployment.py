# Import libraries

import os

from torch.cuda import is_available

import streamlit as st
from streamlit_chat import message

import warnings

from data_ops import DocumentProcessor

from langchain_ollama import ChatOllama
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory

__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

# Ignore all non-failing warnings
warnings.filterwarnings("ignore")

# Page title, page icon, page loading state
st.set_page_config(
    page_title="Education Chatbot",
    page_icon=":parrot:",
    initial_sidebar_state="auto",
    layout="wide",
)

with open("assets/style.css") as css_file:
    css_style = f"<style>{css_file.read()}</style>"

    st.markdown(css_style, unsafe_allow_html=True)

# Create a sidebar on the left
# This sidebar will contain some page styling and information
with st.sidebar:
    st.title("Educational Chatbot")
    st.subheader(
        "This chatbot is designed for educational purposes by the technical team at GritinAI"
    )
    st.image(
        image="./assets/BACKGROUND_IMAGE.jpg",
        width=120,
        use_column_width=True,
    )


########################################
####   DEPLOYMENT IMPLEMENTATION    ####
########################################

MAX_NEW_TOKENS = 512
# MODEL_PATH = "aisquared/dlite-v1-355m"
MODEL_PATH = "openai-community/gpt2"
CONFIG = {"configurable": {"session_id": "abc2"}}
VERSION = "messages"


def generate_prompt_template_v1():
    preamble = """You are a chatbot having a conversation with a human.

    {chat_history}
    """
    template = """Question: {question}

    Answer: Let's think step by step."""

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="You are a chatbot having a conversation with a human."
            ),  # The persistent system prompt
            MessagesPlaceholder(
                variable_name="chat_history"
            ),  # Where the memory will be stored.
            HumanMessagePromptTemplate.from_template(
                "{human_input}"
            ),  # Where the human input will injected
        ]
    )

    return prompt


def generate_prompt_template_from_template():
    template = """Question: {question}

    Answer: Let's think step by step."""

    return ChatPromptTemplate.from_template(template)


def generate_prompt_template_from_messages():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer all questions to the best of your ability.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    return prompt


def generate_prompt_template(version="messages"):
    if version == "messages":
        prompt = generate_prompt_template_from_messages()
    else:
        prompt = generate_prompt_template_from_template()

    return prompt


if "store" not in st.session_state:
    st.session_state.store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = InMemoryChatMessageHistory()
    return st.session_state.store[session_id]


@st.cache_resource
def load_retrieval_system(as_retriever=False, compress_retriever=False):
    """Load trained stenosis detection model"""
    processor = DocumentProcessor(
        as_retriever=as_retriever, compress_retriever=compress_retriever
    )

    return processor


@st.cache_resource
def load_llm(model_name="phi3:mini", temperature=0.5):
    """Load trained stenosis detection model"""

    model = ChatOllama(
        model=model_name,
        temperature=temperature,
    )

    return model


@st.cache_resource
def generate_chain(_model):
    return generate_prompt_template() | _model  # | StrOutputParser()


def generate_final_conversation_chain_v1(llm):
    # llm_chain = generate_chain(_model=llm)
    conversation_with_summary = ConversationChain(
        llm=llm, memory=ConversationSummaryMemory(llm=llm), verbose=True
    )
    return conversation_with_summary


def generate_final_conversation_chain(llm):
    llm_chain = generate_chain(_model=llm)
    stateful_llm_chain = RunnableWithMessageHistory(llm_chain, get_session_history)

    return stateful_llm_chain


def display_chat_history_v1(history=[], user_avatar="personas", ai_avatar=None):
    for i, message_ in enumerate(history[1:]):
        if i % 2 == 0:
            is_user = True
            key = "user_" + str(i)
            avatar_style = user_avatar
        else:
            is_user = False
            key = "ai_" + str(i)
            avatar_style = ai_avatar

        message(message_.content, is_user=is_user, key=key, avatar_style=avatar_style)

    return


def display_chat_history(
    history=[], user_avatar="personas", ai_avatar=None, reverse_history=True
):
    history_ = history[1:]

    if reverse_history:
        history_ = history[::-1]

    chat_pairs = [
        (
            (history_[i + 1], history_[i])
            if reverse_history
            else (history_[i], history_[i + 1])
        )
        for i in range(0, len(history_), 2)
    ]

    for i, (message_1, message_2) in enumerate(chat_pairs):
        user_key = "user_" + str(i)
        ai_key = "ai_" + str(i)

        message(message_1.content, is_user=True, key=user_key, avatar_style=user_avatar)
        message(message_2.content, is_user=False, key=ai_key, avatar_style=ai_avatar)

        st.divider()

    return


def preprocess_documents(documents):
    docs = []
    temp_dir = "./assets/temp"

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    for file in documents:
        temp_filepath = os.path.join(temp_dir, file.name)

        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        docs.append(temp_filepath)

    return docs


def post_process_context(document):
    return document if isinstance(document, str) else document.page_content


def post_process_contexts(predictions):
    if isinstance(predictions, str):
        predictions = [predictions]

    return [post_process_context(document=prediction) for prediction in predictions]


def combine_query_and_contexts(query, contexts):
    if isinstance(contexts, str):
        contexts = [contexts]

    preamble = """Answer the following user request based on the following context. The context will appear first, 
    and then the request.

    {}

    {}
    """

    return [preamble.format(query, context) for context in contexts]


processor = load_retrieval_system()

choices = ["Data Uploads", "Chatbot"]
choice = st.radio(
    label="user_option",
    options=choices,
    index=0,
    horizontal=True,
    label_visibility="hidden",
)


if choice == choices[0]:
    if "processor" not in st.session_state:
        st.session_state[processor] = processor

    # Import image files
    uploaded_files = st.file_uploader(
        "Drag and drop the document(s) here:",
        accept_multiple_files=True,
        type=["pdf", "docx", "csv"],
    )

    print(uploaded_files)

    # if not uploaded_files:
    #     st.info("Please upload documents to continue.")
    #     st.stop()

    uploaded_files = preprocess_documents(uploaded_files)

    print(uploaded_files)

    old_num_chunks = processor.num_chunks

    if uploaded_files:
        st.divider()
        st.write(
            f"Generating vector database... adding to previous {old_num_chunks} documents..."
        )
        st.divider()

    for i, doc in enumerate(uploaded_files, start=1):
        st.write(doc)
        with st.spinner(f"Generating vector database [{i}/{len(uploaded_files)}] ..."):
            split_documents = processor.generate_chunks_from_files(uploaded_files)
            _ = processor.generate_embeddings(documents=split_documents)

    # Persist updated processor
    st.session_state.processor = processor

    if processor.num_chunks > old_num_chunks:
        st.divider()
        st.write("Vector database updated!")
        st.divider()


elif choice == choices[1]:
    # Set the main title of the deployment page
    st.write("""# Educational Chatbot""")
    model_name = st.selectbox(
        label="Select LLM to run",
        options=[
            "phi",
            "phi3:mini",
            "gemma:2b",
            "gemma2:2b",
        ],
        index=1,
    )

    # Pull chosen model to VM via Ollama
    os.system(f"ollama pull {model_name}")

    # Run the code to load the model into memory
    with st.spinner("Model is being loaded.."):
        chatbot_agent = load_retrieval_system()
        llm = load_llm(model_name=model_name)
        llm_chain = generate_chain(_model=llm)
        # llm_memory = configure_chat_memory(llm_chain=llm_chain)

        final_llm_chain = generate_final_conversation_chain(llm=llm)

        DEVICE_ID = 0 if is_available() else -1

        # llm_memory.save_context({"input": "hi"}, {"output": "whats up"})

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            SystemMessage(content="You are a helpful chatbot.")
        ]

    user_query = st.chat_input(placeholder="Ask me anything!", key="user_input")

    if not user_query:
        st.stop()

    # Display user input
    # message(message=user_query, is_user=True, avatar_style="personas")

    # Get context message from RAG retriever
    try:
        results = chatbot_agent.similarity_search(
            query=user_query, mode=0, k=4, top_k=1, total_k=10
        )

        # Postprocess context
        results = post_process_contexts(results)

        if len(results) > 0:
            results = results[0]

        final_query = results + "\n" + user_query
    except:
        st.divider()
        st.write("RAG failed!")
        st.divider()

        results = ""
        final_query = user_query

    # final_query = combine_query_and_contexts(user_query, results)[0]
    # generated_text = llm.invoke(final_query)

    # Append user input to chat history
    human_message = HumanMessage(content=final_query)

    st.session_state["messages"].append(HumanMessage(content=user_query))

    with st.spinner("Generating AI response..."):
        if VERSION == "messages":
            inputs = {"messages": [human_message]}
        else:
            inputs = [human_message]

        generated_output = final_llm_chain.invoke(inputs, config=CONFIG)

    # TODO: Uncomment this for RAG
    # generated_text = generated_output.content.replace(final_query, "")

    generated_text = generated_output.content
    print(generated_output)

    # Write generated text to screen
    # message(generated_text, is_user=False)

    # Append generated output to messages
    st.session_state["messages"].append(AIMessage(content=generated_text))

    history = st.session_state.get("messages", []).copy()

    display_chat_history(history=history, reverse_history=False)
