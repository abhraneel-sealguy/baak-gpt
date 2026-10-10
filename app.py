import streamlit as st
import ollama
import pandas as pd
from pypdf import PdfReader
from PIL import Image
import io
import torch
from diffusers import AutoPipelineForText2Image

# App Configuration
st.set_page_config(page_title="Baak GPT Pro", page_icon="🤖", layout="wide")

# Change this line near the top of your app.py script:
VISION_MODEL = "llava"

# --- FRIENDLY SYSTEM PROMPT ---
# This gives the AI its personality, making it act like an empathetic peer/friend.
FRIENDLY_SYSTEM_PROMPT = (
    "You are Baak GPT, a warm, highly capable, and adaptive AI companion. "
    "When the user is just chatting, act like an empathetic, supportive peer. Use casual, "
    "approachable, and universal language. Avoid being overly formal or robotic. Match the user's energy, "
    "and feel free to use light humor or friendly expressions. However, when the user uploads a document, "
    "an Excel sheet, or a math/physics image, seamlessly transition into a brilliant, structured expert "
    "to solve their problems step-by-step with absolute precision."
)

# --- CACHED IMAGE GENERATION PIPELINE ---
@st.cache_resource
def load_image_generator():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            variant="fp16" if device == "cuda" else None
        )
        pipe.to(device)
        return pipe
    except Exception as e:
        return None

# --- UI SIDEBAR FOR MODES ---
with st.sidebar:
    st.title("⚙️ Baak GPT Control Panel")
    mode = st.radio("Select Capability Mode:", ["Chat & Document Analysis", "Paint Studio (Offline TTI)"])
    st.markdown("---")
    st.info("💡 **Tip:** In Chat Mode, you can just talk freely about your day, or upload images, PDFs, and spreadsheets to analyze them.")

# ==========================================
# MODE 1: PAINT STUDIO (IMAGE GENERATION)
# ==========================================
if mode == "Paint Studio (Offline TTI)":
    st.header("🎨 Baak GPT Paint Studio")
    st.caption("Generate high-quality artwork entirely on your hard drive with zero internet.")
    
    image_prompt = st.text_area("Describe the image you want to create:")
    steps = st.slider("Quality Steps", min_value=1, max_value=10, value=2)
    
    if st.button("✨ Generate Image"):
        if image_prompt:
            with st.spinner("Generating pixels locally... Please wait."):
                generator_pipe = load_image_generator()
                if generator_pipe:
                    generated_img = generator_pipe(prompt=image_prompt, num_inference_steps=steps, guidance_scale=0.0).images
                    st.image(generated_img, caption="Generated completely offline by Baak GPT", use_container_width=True)
                else:
                    st.error("Could not load local Image Generator. Ensure your machine has enough RAM/VRAM.")
        else:
            st.warning("Please type a description first!")

# ==========================================
# MODE 2: FRIENDLY CHAT & MULTIMODAL WORKSPACE
# ==========================================
else:
    st.header("🤖 Baak GPT Workspace")
    st.caption("Talk freely like a friend, or drop files to solve complex technical tasks.")

    # Initialize chat history with system prompt tracking
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hey there! I am Baak GPT. We can just hang out and talk, or you can drop a complex file below for me to crack open. What's on your mind?"}
        ]

    # File Upload Widget
    uploaded_file = st.file_uploader("Upload Image, PDF, or Excel spreadsheet:", type=["png", "jpg", "jpeg", "pdf", "xlsx", "xls"])

    # Display History to UI
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Process Input
    if user_input := st.chat_input("Message Baak GPT..."):
        
        # Display user message instantly
        with st.chat_message("user"):
            st.markdown(user_input)

        image_bytes = None
        injected_context = ""

        if uploaded_file is not None:
            file_type = uploaded_file.name.split(".")[-1].lower()
            
            # --- 1. HANDLE IMAGES (Resolves RGBA / transparency errors) ---
            if file_type in ["png", "jpg", "jpeg"]:
                img = Image.open(uploaded_file)
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")  # Safely flattens alpha channel transparency
                
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG")
                image_bytes = buffered.getvalue()
                injected_context = "[User attached an image file for you to analyze or solve]\n"

            # --- 2. HANDLE PDFs ---
            elif file_type == "pdf":
                with st.spinner("Reading PDF pages..."):
                    reader = PdfReader(uploaded_file)
                    pdf_text = "".join([page.extract_text() + "\n" for page in reader.pages])
                    injected_context = f"=== CONTEXT FROM PDF ({uploaded_file.name}) ===\n{pdf_text}\n====================\n"

            # --- 3. HANDLE EXCEL ---
            elif file_type in ["xlsx", "xls"]:
                with st.spinner("Formatting spreadsheets..."):
                    df = pd.read_excel(uploaded_file)
                    injected_context = f"=== CONTEXT FROM EXCEL ({uploaded_file.name}) ===\n{df.to_markdown(index=False)}\n====================\n"

        # Final context consolidation
        current_prompt = injected_context + user_input
        
        # Build prompt dictionary payload, beginning with hidden personality structure
        ollama_messages = [{"role": "system", "content": FRIENDLY_SYSTEM_PROMPT}]
        
        for msg in st.session_state.messages:
            ollama_messages.append({"role": msg["role"], "content": msg["content"]})
            
        latest_payload = {"role": "user", "content": current_prompt}
        if image_bytes:
            latest_payload["images"] = [image_bytes]
            
        ollama_messages.append(latest_payload)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Safe streaming output
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            try:
                response_stream = ollama.chat(
                    model=VISION_MODEL,
                    messages=ollama_messages,
                    stream=True
                )
                
                for chunk in response_stream:
                    content = chunk['message']['content']
                    full_response += content
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                response_placeholder.markdown(f"❌ **Ollama Backend Error:** {str(e)}\n\nIf you see an 'mllama' error, please fully exit Ollama from your Windows system tray near the clock, restart it, and try again.")
