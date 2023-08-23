from typing import Optional

import openai
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage

from prompts import system_prompt, user_information




def get_cover_letter_langchain_chat_prompts(
        resume: str,
        job_description: str,
        openai_api_key: str,
        model: str,
        additional_information: Optional[str] = None
) -> str:
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt)
    user_message_prompt = HumanMessagePromptTemplate.from_template(user_information)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, user_message_prompt]
    )

    llm = ChatOpenAI(temperature=0.2, openai_api_key=openai_api_key, model=model)
    llm_chain = LLMChain(
        llm=llm,
        prompt=chat_prompt
    )
    additional_information = additional_information if additional_information is not None else "None"
    args = {
        "resume": resume,
        "job_description": job_description,
        "additional_information": additional_information
    }

    return llm_chain.run(args)

