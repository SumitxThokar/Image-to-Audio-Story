from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFaceHub
import requests
import os
import streamlit as st

load_dotenv(find_dotenv())

# image to text
def img2txt(url):
    pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base",max_new_tokens=8)
    text = pipe(url)[0]["generated_text"]
    print(text)
    return text

# llm
def generate_story(scenario):
    template = """You are william shakespeare. Generate a description based on a given context below, the description should be no more than 20 words
      CONTEXT:{scenario}
        """
    prompt = PromptTemplate(template=template, input_variables=["scenario"])

    story_llm = LLMChain(llm=HuggingFaceHub(
        repo_id="bigscience/bloom",
        model_kwargs={"temperature": 1, "max_length": 30}
    ), prompt=prompt)
    
    story = story_llm.predict(scenario=scenario)
    print(story)
    return story


def txt2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": "Bearer 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"}
    payloads = {
        "inputs":message
    }
    response = requests.post(API_URL, headers=headers, json=payloads)
    with open('audio.flac','wb') as file:
        file.write(response.content)


def main():
    st.set_page_config(page_title="Image to Audio Story", page_icon="🚀")

    st.header("Turn Your Image into Audio Story.")
    uploaded_file = st.file_uploader("Choose an image..", type="jpg")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption= 'Image Uploaded.',
                 use_column_width=True)
        scenario = img2txt(uploaded_file.name)
        story = generate_story(scenario)
        txt2speech(story)
        with st.expander("Scenario"):
            st.write(scenario)
        with st.expander("Story"):
            st.write(story)
        st.audio("audio.flac")

if __name__=='__main__':
    main()
