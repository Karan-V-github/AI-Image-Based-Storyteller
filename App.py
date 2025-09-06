import os
import streamlit as st
from dotenv import load_dotenv
from transformers import pipeline

# Load Hugging Face token
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACEHUB_API_KEY")
if not HF_TOKEN:
    st.error("⚠️ Missing Hugging Face API token. Add it to your .env file as HUGGINGFACEHUB_API_TOKEN.")
    st.stop()

# ---------- IMAGE TO TEXT ----------
def imagetotext(img_path):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    description = image_to_text(img_path)[0]["generated_text"]
    print("Detected content:", description)
    return description

# ---------- DESCRIPTION TO STORY ----------
def generate_story(description):
    """Generate a creative story from image description using Flan-T5 with few-shot examples."""
    story_generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-large"
    )

    # Few-shot prompt
    prompt = f"""
You are a creative storyteller. 
You are given a short image description. 
Write a detailed, imaginative, and engaging story of about 150 words.
Include a beginning, middle, and end. Use varied sentences and imagination. 
Do NOT repeat the description word-for-word.

Example:
Description: A cat wearing a wizard hat.
Story: Once upon a time, a clever cat discovered a hidden spellbook in the attic. It practiced magic spells every night, causing small mysterious events around the house. One day, the cat accidentally summoned a tiny dragon, and together they embarked on magical adventures that taught them courage, friendship, and the importance of curiosity.

Now, generate a story for this description:
Description: {description}
Story:
"""

    story = story_generator(
        prompt,
        max_new_tokens=400,
        temperature=0.9,
        top_p=0.95,
        do_sample=True
    )[0]["generated_text"]

    return story.strip()

# ---------- STREAMLIT UI ----------
st.title("📖 Image to Story Generator")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    with open("uploaded.png", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image("uploaded.png", caption="Uploaded Image", use_column_width=True)

    with st.spinner("🔎 Analyzing image..."):
        caption = imagetotext("uploaded.png")
    st.write("**Detected content:**", caption)

    with st.spinner("✍️ Writing story..."):
        story = generate_story(caption)
    st.subheader("Generated Story")
    st.write(story)
