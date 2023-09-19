# Image to Audio Story with Langchain

This is a beginner-friendly tutorial on how to use Langchain to turn an image into an audio story.

## Requirements:

- A Langchain account
- An image to convert
- A text-to-speech API key
## Steps:

1. Install the required dependencies:
  ```shell
pip install from-dotenv transformers langchain streamlit
  ```
2. Create a .env file and add your Langchain API key:
```shell
# .env
LANGCHAIN_API_KEY=YOUR_API_KEY
```
3. Load the .env file:
```shell
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())
```
4. Define a function to convert an image to text:
```Python
def img2txt(url):
    pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base",max_new_tokens=8)
    text = pipe(url)[0]["generated_text"]
    print(text)
    return text
```
5. Define a function to generate a story from a scenario:
```Python
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
```
6. Define a function to convert text to speech:
```Python
def txt2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": "Bearer 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"}
    payloads = {
        "inputs":message
    }
    response = requests.post(API_URL, headers=headers, json=payloads)
    with open('audio.flac','wb') as file:
        file.write(response.content)
```
7. Write the main function:
```Python
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
```
8. Run the app:
```shell
streamlit run app.py
```
Click the "Choose an image.." button to upload your image. Once the image is uploaded, the app will generate a scenario and a story from it. You can then listen to the story by clicking the "Play" button.

## Tips:

Try using different images to see what kind of stories the AI can generate.
You can also use the "Scenario" and "Story" expanders to edit the scenario and story before converting it to speech.
If you are not satisfied with the results, try changing the temperature and max_length parameters in the generate_story() function.

