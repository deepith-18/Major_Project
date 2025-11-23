import streamlit as st
import torch
import os
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from PIL import Image

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Fake Review Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PROFESSIONAL LOOK ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stButton>button { 
        width: 100%; 
        background-color: #FF4B4B; 
        color: white; 
        font-weight: bold; 
        border-radius: 8px;
        height: 50px;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LABEL MAPPING ---
# Based on your test: Label 2 was the Spam/Fake one.
# You can swap 0 and 1 if you find "Real" reviews are showing as "Neutral".
LABEL_MAP = {
    0: "GENUINE REVIEW",   # Likely Real
    1: "NEUTRAL / UNSURE", # Likely Middle ground
    2: "FAKE / SPAM"       # Confirmed by your screenshot
}

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    model_path = "models/distilbert_pytorch_model.pth"
    tokenizer_path = "models/distilbert_pytorch_tokenizer"

    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        st.error("❌ System Error: Model files missing.")
        st.stop()

    try:
        tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=3
        )
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Model Load Error: {e}")
        st.stop()

# --- INITIALIZE ---
tokenizer, model = load_model()

# --- SIDEBAR ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2092/2092663.png", width=100)
st.sidebar.title("Control Panel")
app_mode = st.sidebar.radio("Select Mode", ["🕵️‍♂️ Live Analysis", "📊 Performance Metrics"])
st.sidebar.divider()
st.sidebar.info(
    "**System Info**\n"
    "- Model: DistilBERT Transformer\n"
    "- Accuracy: ~92% (Training)\n"
    "- Backend: PyTorch"
)

# --- TAB 1: LIVE ANALYSIS ---
if app_mode == "🕵️‍♂️ Live Analysis":
    st.title("🛡️ Fake Review Detection System")
    st.markdown("### 7th Semester Major Project")
    st.markdown("Enter a product review below to detect if it is **Genuine** or **Computer-Generated/Spam**.")

    col1, col2 = st.columns([2, 1])

    with col1:
        user_input = st.text_area("Input Review:", height=200, placeholder="Paste the review text here...")
        
        if st.button("🔍 Analyze Review"):
            if not user_input.strip():
                st.warning("⚠️ Please enter text to analyze.")
            else:
                with st.spinner("Processing with DistilBERT..."):
                    # Tokenize & Predict
                    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                    
                    # Get Result
                    class_id = torch.argmax(probs).item()
                    confidence = probs[0][class_id].item()
                    result_text = LABEL_MAP.get(class_id, "Unknown")

                  # Color Logic (UPDATED)
                    if class_id == 0: # Fake (Class 0)
                        color = "#ff4b4b" # Red
                        icon = "🚨"
                    elif class_id == 2: # Genuine (Class 2)
                        color = "#2ecc71" # Green
                        icon = "✅"
                    else:
                        color = "#f1c40f" # Yellow
                        icon = "⚠️"

                # Display Result
                st.markdown(f"""
                    <div class="result-card" style="background-color: {color}20; border: 2px solid {color};">
                        <h1 style="color: {color}; margin:0;">{icon} {result_text}</h1>
                        <p style="font-size: 20px; margin:0;">Confidence Score: <b>{confidence*100:.2f}%</b></p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.write("")
                st.write("### Probability Breakdown")
                # Show bar chart of all 3 probabilities
                chart_data = {
                    "Genuine": probs[0][0].item(),
                    "Neutral": probs[0][1].item(),
                    "Fake": probs[0][2].item()
                }
                st.bar_chart(chart_data)

    with col2:
        st.subheader("How it works")
        st.markdown("""
        1. **Input:** User enters raw text.
        2. **Tokenization:** Text is converted into numerical vectors using the BERT tokenizer.
        3. **Inference:** The DistilBERT model analyzes semantic context.
        4. **Output:** The system classifies the review based on learned patterns.
        """)

# --- TAB 2: METRICS ---
elif app_mode == "📊 Performance Metrics":
    st.title("📊 Model Training Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Training Accuracy")
        if os.path.exists("images/pytorch_accuracy.png"):
            st.image("images/pytorch_accuracy.png", use_container_width=True)
            st.caption("Accuracy increases as the model learns over epochs.")
        else:
            st.warning("Accuracy graph missing.")

    with col2:
        st.subheader("Training Loss")
        if os.path.exists("images/pytorch_loss.png"):
            st.image("images/pytorch_loss.png", use_container_width=True)
            st.caption("Loss decreases as the model minimizes errors.")
        else:
            st.warning("Loss graph missing.")