# Import libraries

import os

import getpass
import os

from dotenv import load_dotenv

# if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
#     os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass("Enter your token: ")


load_dotenv(dotenv_path='.env')

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline

from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

from transformers import pipeline
import torch


__all__ = [
    "load_language_model",
    "generate_generation_chain",
    "generate_predictions",
]


def load_llm(model_path="aisquared/dlite-v1-355m", max_new_tokens=1024, **kwargs):
    text_generator = pipeline(
        task="text-generation",
        model=model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        framework="pt",
        max_new_tokens=max_new_tokens,
        **kwargs,
    )

    return text_generator


# Define a function to load model into memory and cache it
def load_language_model(
    model_path="aisquared/dlite-v1-355m",
    raw_pipe=False,
    max_new_tokens=1024,
    device=-1,
    repetition_penalty=1.03,
    **kwargs,
):
    """Load trained stenosis detection model"""
    if max_new_tokens is None:
        max_new_tokens = 50

    if raw_pipe:
        return load_llm(model_path=model_path, **kwargs)
    else:
        return HuggingFaceEndpoint(
            repo_id=model_path,
            task="text-generation",
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=repetition_penalty,
            device=device,
            **kwargs,
        )


def generate_generation_chain(model, skip_prompt=False):
    template = """Input: {user_query}
    
    Answers:
    
    """

    prompt = PromptTemplate.from_template(template)

    # llm_chain = LLMChain(llm=HuggingFacePipeline(pipeline=model), prompt=prompt)
    if skip_prompt:
        llm_chain = prompt | HuggingFacePipeline(pipeline=model).bind(
            skip_prompt=skip_prompt
        )
    else:
        llm_chain = prompt | HuggingFacePipeline(pipeline=model)

    return llm_chain


def generate_predictions(user_query, llm_chain):
    return llm_chain.invoke({"user_query": user_query})


def run_text_generation_pipeline(model, user_query, use_chain=False):
    if use_chain:
        chain = generate_generation_chain(model=model)
        # result = chain.run(user_query)
        result = generate_predictions(user_query=user_query, llm_chain=chain)
    else:
        result = model(user_query)

    return result


if __name__ == "__main__":
    MAX_NEW_TOKENS = 100

    query = """What do the capital of Japan and Game of Thrones have in common?"""
    text = query[::1]
    model_path = "aisquared/dlite-v1-355m"

    llm = load_language_model(model_path=model_path)
    # wrapped_llm = HuggingFacePipeline(pipeline=llm)

    wrapped_llm = HuggingFacePipeline.from_model_id(
        model_id=model_path,
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": MAX_NEW_TOKENS},
        device=0,
    )
    # wrapped_llm = HuggingFacePipeline(pipeline=llm)
    #
    # print(wrapped_llm)
    #
    # # try:
    # chain = generate_generation_chain(model=wrapped_llm)
    # # result = chain.run(query)
    #
    # print(dir(chain))
    # result = generate_predictions(user_query=query, llm_chain=chain)
    # # except:
    # #     result = llm(query)
    #
    # print(result)

    question_1 = "What NFL team won the Super Bowl in the year Justin Beiber was born?"
    question_2 = "Which actor played Jon Snow in the Game of Thrones tv series?"
    question_3 = f"""
    ### System:
    You are an AI assistant that follows instruction extremely well. 
    Help as much as you can. Please be truthful and give direct answers

    ### User:
    {query}

    ### Response:
    """

    template = f"""Question: {text}
    Answer: Let's think step by step."""

    # Uncomment the code below to try the langchain Chain
    prompt = PromptTemplate(template=template, input_variables=["text"])
    prompt_ = PromptTemplate.from_template(template=template)

    # llm_chain = LLMChain(prompt=prompt, llm=llm)
    llm_chain = prompt_ | wrapped_llm

    answers = run_text_generation_pipeline(llm, question_2, use_chain=False)

    print("=" * 50)
    print(answers)

    answers_ = wrapped_llm.invoke(question_2)
    print("=" * 50)
    print(answers_)
