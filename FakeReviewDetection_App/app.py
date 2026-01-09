import streamlit as st
import torch
import os
import time
import json
import pandas as pd
from datetime import datetime
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import plotly.graph_objects as go

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'history' not in st.session_state:
    st.session_state.history = []
    
if 'stats' not in st.session_state:
    st.session_state.stats = {'total': 0, 'fake': 0, 'genuine': 0}
    
if 'model_info' not in st.session_state:
    st.session_state.model_info = {}

if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

if 'confidence_threshold' not in st.session_state:
    st.session_state.confidence_threshold = 0.5

if 'show_animations' not in st.session_state:
    st.session_state.show_animations = True

if 'auto_save_history' not in st.session_state:
    st.session_state.auto_save_history = True

if 'comparison_reviews' not in st.session_state:
    st.session_state.comparison_reviews = []

# ============================================================================
# THEME SYSTEM
# ============================================================================

def get_theme_colors():
    """Return colors based on current theme"""
    if st.session_state.theme == 'dark':
        return {
            'page_bg': '#0e1117',
            'sidebar_bg': '#262730',
            'bg_gradient': 'linear-gradient(135deg, #0e1117 0%, #1a1d29 100%)',
            'card_bg': '#1e2130',
            'card_hover': '#2a2d3e',
            'text_primary': '#ffffff',
            'text_secondary': '#cccccc',
            'border': '#444',
            'shadow': 'rgba(0, 0, 0, 0.3)',
            'input_bg': '#262730',
            'success_bg': '#1a4d2e',
            'warning_bg': '#4d3319',
            'error_bg': '#4d1919'
        }
    else:
        return {
            'page_bg': '#f5f7fa',
            'sidebar_bg': '#e8ecf0',
            'bg_gradient': 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
            'card_bg': '#ffffff',
            'card_hover': '#f8f9fa',
            'text_primary': '#2c3e50',
            'text_secondary': '#5a6c7d',
            'border': '#dee2e6',
            'shadow': 'rgba(0, 0, 0, 0.1)',
            'input_bg': '#f8f9fa',
            'success_bg': '#d4edda',
            'warning_bg': '#fff3cd',
            'error_bg': '#f8d7da'
        }

colors = get_theme_colors()

# ============================================================================
# CUSTOM CSS WITH FULL PAGE THEME SUPPORT
# ============================================================================

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    * {{ 
        font-family: 'Poppins', sans-serif; 
    }}
    
    /* Full Page Background */
    .stApp {{
        background: {colors['bg_gradient']} !important;
    }}
    
    /* Main Content Area */
    .main {{
        background: transparent !important;
        color: {colors['text_primary']} !important;
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: {colors['sidebar_bg']} !important;
    }}
    
    [data-testid="stSidebar"] > div:first-child {{
        background: {colors['sidebar_bg']} !important;
    }}
    
    /* Text Elements */
    .stMarkdown, p, span, div, label {{
        color: {colors['text_primary']} !important;
    }}
    
    h1, h2, h3, h4, h5, h6 {{
        color: {colors['text_primary']} !important;
    }}
    
    /* Text Input & Text Area */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {{
        background-color: {colors['input_bg']} !important;
        color: {colors['text_primary']} !important;
        border: 1px solid {colors['border']} !important;
    }}
    
    /* Select Box */
    .stSelectbox > div > div > div {{
        background-color: {colors['input_bg']} !important;
        color: {colors['text_primary']} !important;
    }}
    
    /* Slider */
    .stSlider > div > div > div > div {{
        background-color: {colors['card_bg']} !important;
    }}
    
    /* Expander */
    .streamlit-expanderHeader {{
        background-color: {colors['card_bg']} !important;
        color: {colors['text_primary']} !important;
        border: 1px solid {colors['border']} !important;
    }}
    
    .streamlit-expanderContent {{
        background-color: {colors['card_bg']} !important;
        border: 1px solid {colors['border']} !important;
    }}
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {{
        color: {colors['text_primary']} !important;
    }}
    
    /* Info/Warning/Error Boxes */
    .stAlert {{
        background-color: {colors['card_bg']} !important;
        color: {colors['text_primary']} !important;
        border: 1px solid {colors['border']} !important;
    }}
    
    /* Animated Header */
    .animated-header {{
        background: linear-gradient(45deg, #FF4B4B, #FF6B6B);
        padding: 30px; 
        border-radius: 15px; 
        text-align: center; 
        margin-bottom: 30px;
        box-shadow: 0 10px 40px rgba(255, 75, 75, 0.3); 
        color: white !important;
        animation: fadeInDown 0.8s ease-out;
    }}
    
    .animated-header h1, .animated-header p {{
        color: white !important;
    }}
    
    @keyframes fadeInDown {{
        from {{
            opacity: 0;
            transform: translateY(-20px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    /* Feature Cards */
    .feature-card {{
        background: {colors['card_bg']} !important; 
        padding: 20px; 
        border-radius: 12px;
        margin-bottom: 15px; 
        border-left: 5px solid #FF4B4B;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        color: {colors['text_primary']} !important;
        box-shadow: 0 2px 8px {colors['shadow']};
    }}
    
    .feature-card:hover {{
        transform: translateX(5px);
        box-shadow: 0 5px 20px rgba(255, 75, 75, 0.2);
        background: {colors['card_hover']} !important;
    }}
    
    .feature-card * {{
        color: {colors['text_primary']} !important;
    }}
    
    /* Stat Cards */
    .stat-card {{
        background: {colors['card_bg']} !important;
        padding: 20px; 
        border-radius: 10px;
        text-align: center; 
        border: 1px solid {colors['border']};
        transition: transform 0.3s ease;
        color: {colors['text_primary']} !important;
        box-shadow: 0 2px 8px {colors['shadow']};
    }}
    
    .stat-card:hover {{
        transform: scale(1.05);
    }}
    
    .stat-card * {{
        color: {colors['text_primary']} !important;
    }}
    
    /* Buttons */
    .stButton>button {{
        background: linear-gradient(45deg, #FF4B4B, #FF6B6B) !important;
        color: white !important; 
        border: none !important; 
        height: 50px; 
        font-size: 1.1em;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.3);
    }}
    
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.4);
    }}
    
    /* Download Button */
    .stDownloadButton>button {{
        background: linear-gradient(45deg, #3498db, #2980b9) !important;
        color: white !important;
    }}
    
    /* Pulse Animation */
    .pulse {{
        animation: pulse 2s infinite;
    }}
    
    @keyframes pulse {{
        0% {{
            box-shadow: 0 0 0 0 rgba(255, 75, 75, 0.7);
        }}
        70% {{
            box-shadow: 0 0 0 10px rgba(255, 75, 75, 0);
        }}
        100% {{
            box-shadow: 0 0 0 0 rgba(255, 75, 75, 0);
        }}
    }}
    
    /* Slide In Animation */
    @keyframes slideIn {{
        from {{
            opacity: 0;
            transform: translateX(-20px);
        }}
        to {{
            opacity: 1;
            transform: translateX(0);
        }}
    }}
    
    .slide-in {{
        animation: slideIn 0.5s ease-out;
    }}
    
    /* Progress Bar */
    .stProgress > div > div > div > div {{
        background: linear-gradient(45deg, #FF4B4B, #FF6B6B) !important;
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: {colors['card_bg']} !important;
        border-radius: 10px;
        padding: 5px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        color: {colors['text_primary']} !important;
        border-radius: 8px;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(45deg, #FF4B4B, #FF6B6B) !important;
        color: white !important;
    }}
    
    /* Divider */
    hr {{
        border-color: {colors['border']} !important;
    }}
    
    /* Code Blocks */
    code {{
        background-color: {colors['card_bg']} !important;
        color: {colors['text_primary']} !important;
        border: 1px solid {colors['border']} !important;
    }}
    
    /* DataFrames */
    .dataframe {{
        background-color: {colors['card_bg']} !important;
        color: {colors['text_primary']} !important;
    }}
    
    /* Comparison Card */
    .comparison-card {{
        background: {colors['card_bg']};
        border: 2px solid {colors['border']};
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        transition: all 0.3s ease;
    }}
    
    .comparison-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 25px {colors['shadow']};
    }}
    
    /* Badge */
    .badge {{
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.9em;
        font-weight: 600;
        margin: 5px;
    }}
    
    .badge-fake {{
        background: #e74c3c;
        color: white;
    }}
    
    .badge-genuine {{
        background: #2ecc71;
        color: white;
    }}
    
    /* Loading Spinner */
    .stSpinner > div {{
        border-top-color: #FF4B4B !important;
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL LOADING WITH PROPER VALIDATION
# ============================================================================

@st.cache_resource
def load_backend_model():
    """
    Load the trained DistilBERT model with proper validation.
    Returns: tokenizer, model, num_labels, label_mapping
    """
    model_path = os.path.join("models", "distilbert_pytorch_model.pth")
    tokenizer_path = os.path.join("models", "distilbert_pytorch_tokenizer")
    
    if not os.path.exists(model_path):
        st.error(f"❌ CRITICAL ERROR: Model file not found at {model_path}")
        st.info("💡 Please ensure your trained model is saved at the correct path.")
        st.stop()
    
    if not os.path.exists(tokenizer_path):
        st.error(f"❌ CRITICAL ERROR: Tokenizer not found at {tokenizer_path}")
        st.stop()
    
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        classifier_weight_shape = state_dict['classifier.weight'].shape
        num_labels = classifier_weight_shape[0]
        
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", 
            num_labels=num_labels
        )
        
        model.load_state_dict(state_dict)
        model.eval()
        
        if num_labels == 2:
            label_map = {0: "FAKE", 1: "GENUINE"}
        elif num_labels == 3:
            label_map = {0: "FAKE", 1: "NEUTRAL", 2: "GENUINE"}
        else:
            label_map = {i: f"CLASS_{i}" for i in range(num_labels)}
        
        return tokenizer, model, num_labels, label_map
        
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        st.exception(e)
        st.stop()

# Initialize model
with st.spinner("🔄 Loading AI Model..."):
    tokenizer, model, num_labels, LABEL_MAP = load_backend_model()
    st.session_state.model_info = {
        'num_labels': num_labels,
        'label_map': LABEL_MAP,
        'loaded_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# ============================================================================
# CORE PREDICTION FUNCTION (PRESERVED)
# ============================================================================

def analyze_review(text):
    """
    Analyze a single review using the trained model.
    """
    if not text or len(text.strip()) == 0:
        return "INVALID", 0.0, None
    
    try:
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            predicted_class_id = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class_id].item()
            predicted_label = LABEL_MAP.get(predicted_class_id, "UNKNOWN")
            
        return predicted_label, confidence, probabilities[0]
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return "ERROR", 0.0, None

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def export_history_json():
    """Export history as JSON"""
    return json.dumps(st.session_state.history, indent=2)

def export_history_csv():
    """Export history as CSV"""
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        return df.to_csv(index=False)
    return ""

def create_gauge_chart(confidence, result):
    """Create an animated gauge chart for confidence"""
    color = "#e74c3c" if "FAKE" in result else "#2ecc71"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence", 'font': {'size': 24, 'color': colors['text_primary']}},
        delta={'reference': 50, 'increasing': {'color': color}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': colors['text_secondary']},
            'bar': {'color': color},
            'bgcolor': colors['card_bg'],
            'borderwidth': 2,
            'bordercolor': colors['border'],
            'steps': [
                {'range': [0, 50], 'color': 'rgba(231, 76, 60, 0.2)'},
                {'range': [50, 100], 'color': 'rgba(46, 204, 113, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': st.session_state.confidence_threshold * 100
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(t=50, b=20, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': colors['text_primary'], 'family': "Poppins"}
    )
    
    return fig

def create_history_chart():
    """Create a timeline chart of analysis history"""
    if not st.session_state.history:
        return None
    
    df = pd.DataFrame(st.session_state.history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    fake_counts = []
    genuine_counts = []
    timestamps = []
    
    for i in range(len(df)):
        subset = df.iloc[:i+1]
        fake_counts.append(len(subset[subset['result'] == 'FAKE']))
        genuine_counts.append(len(subset[subset['result'] == 'GENUINE']))
        timestamps.append(subset.iloc[-1]['timestamp'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=fake_counts,
        mode='lines+markers',
        name='Fake Reviews',
        line=dict(color='#e74c3c', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=genuine_counts,
        mode='lines+markers',
        name='Genuine Reviews',
        line=dict(color='#2ecc71', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Analysis Trend Over Time",
        xaxis_title="Time",
        yaxis_title="Count",
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': colors['text_primary'], 'family': "Poppins"},
        hovermode='x unified',
        xaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)')
    )
    
    return fig

def auto_save_history():
    """Auto-save history to JSON file"""
    if st.session_state.auto_save_history and st.session_state.history:
        try:
            filename = "auto_save_history.json"
            with open(filename, 'w') as f:
                json.dump(st.session_state.history, f, indent=2)
            return True
        except:
            return False
    return False

# ============================================================================
# UI HEADER WITH THEME TOGGLE
# ============================================================================

col_header, col_toggle = st.columns([5, 1])

with col_header:
    st.markdown('''
    <div class="animated-header">
        <h1>🛡️ Fake Review Detector</h1>
        <p style="font-size: 1.2em; margin: 10px 0 0 0;">
            Powered by DistilBERT Neural Network
        </p>
        <p style="font-size: 0.9em; opacity: 0.9; margin: 5px 0 0 0;">
            Advanced AI Detection System
        </p>
    </div>
    ''', unsafe_allow_html=True)

with col_toggle:
    st.markdown("<br>", unsafe_allow_html=True)
    theme_icon = "☀️" if st.session_state.theme == 'dark' else "🌙"
    if st.button(f"{theme_icon} Theme", use_container_width=True, key="theme_toggle_main"):
        st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'
        st.rerun()

# ============================================================================
# SIDEBAR (ENHANCED)
# ============================================================================

with st.sidebar:
    st.title("⚙️ Control Panel")
    
    # Navigation
    app_mode = st.selectbox(
        "📍 Navigate",
        ["🔍 Single Analysis", "📊 Batch Analysis", "🔬 Compare Reviews", "📜 History", "📈 Analytics", "🤖 Model Info", "⚙️ Settings"]
    )
    
    st.divider()
    
    # Statistics
    st.markdown("### 📊 Session Statistics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'''
        <div class="stat-card">
            <h2 style="margin: 0; color: #3498db;">{st.session_state.stats['total']}</h2>
            <p style="margin: 5px 0 0 0; font-size: 0.9em;">Total</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="stat-card">
            <h2 style="margin: 0; color: #e74c3c;">{st.session_state.stats['fake']}</h2>
            <p style="margin: 5px 0 0 0; font-size: 0.9em;">Fake</p>
        </div>
        ''', unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    with col3:
        st.markdown(f'''
        <div class="stat-card">
            <h2 style="margin: 0; color: #2ecc71;">{st.session_state.stats['genuine']}</h2>
            <p style="margin: 5px 0 0 0; font-size: 0.9em;">Genuine</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        accuracy = (st.session_state.stats['genuine'] / st.session_state.stats['total'] * 100) if st.session_state.stats['total'] > 0 else 0
        st.markdown(f'''
        <div class="stat-card">
            <h2 style="margin: 0; color: #f39c12;">{accuracy:.1f}%</h2>
            <p style="margin: 5px 0 0 0; font-size: 0.9em;">Genuine Rate</p>
        </div>
        ''', unsafe_allow_html=True)
    
    st.divider()
    
    # Quick Actions
    st.markdown("### ⚡ Quick Actions")
    
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        if st.button("📥 JSON", use_container_width=True, help="Export as JSON"):
            if st.session_state.history:
                st.download_button(
                    label="⬇️ Download",
                    data=export_history_json(),
                    file_name=f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True,
                    key="download_json_sidebar"
                )
            else:
                st.warning("No history")
    
    with col_exp2:
        if st.button("📥 CSV", use_container_width=True, help="Export as CSV"):
            if st.session_state.history:
                st.download_button(
                    label="⬇️ Download",
                    data=export_history_csv(),
                    file_name=f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_csv_sidebar"
                )
            else:
                st.warning("No history")
    
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    
    if st.button("🔄 Reset Stats", use_container_width=True):
        st.session_state.stats = {'total': 0, 'fake': 0, 'genuine': 0}
        st.rerun()
    
    # Auto-save indicator
    if st.session_state.auto_save_history:
        st.markdown("---")
        st.markdown("💾 **Auto-save:** ON")

# ============================================================================
# PAGE 1: SINGLE REVIEW ANALYSIS (ENHANCED)
# ============================================================================

if app_mode == "🔍 Single Analysis":
    st.markdown("## 🔍 Single Review Detection")
    st.markdown("Enter a product review below to check its authenticity using AI.")
    
    # Sample reviews for testing
    with st.expander("📝 Try Sample Reviews"):
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Sample: Likely Fake", use_container_width=True):
                st.session_state.sample_text = "This product is absolutely amazing!!! Best thing ever!!! Everyone must buy this incredible amazing wonderful product!!!"
        with col2:
            if st.button("Sample: Likely Genuine", use_container_width=True):
                st.session_state.sample_text = "Good product overall. The quality is decent and it works as described. Shipping took 3 days. Would recommend for the price."
        with col3:
            if st.button("Sample: Mixed", use_container_width=True):
                st.session_state.sample_text = "The product works fine but took forever to arrive. Customer service was helpful though. Price is a bit high for what you get."
    
    # Text input
    review_text = st.text_area(
        "Enter Review Text:",
        height=150,
        placeholder="Type or paste the review here...",
        value=st.session_state.get('sample_text', ''),
        key="review_input"
    )
    
    # Options
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        add_to_comparison = st.checkbox("➕ Add to comparison", value=False)
    with col_opt2:
        show_detailed_analysis = st.checkbox("📊 Show detailed analysis", value=True)
    
    # Analyze button
    if st.button("🔍 Analyze Review", use_container_width=True, type="primary"):
        if review_text.strip():
            with st.spinner("🤖 AI is analyzing the review..."):
                time.sleep(0.5) if st.session_state.show_animations else time.sleep(0.1)
                result, confidence, probabilities = analyze_review(review_text)
                
                st.session_state.stats['total'] += 1
                if "FAKE" in result:
                    st.session_state.stats['fake'] += 1
                elif "GENUINE" in result:
                    st.session_state.stats['genuine'] += 1
                
                analysis_record = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'text': review_text,
                    'result': result,
                    'confidence': confidence
                }
                
                st.session_state.history.append(analysis_record)
                
                if add_to_comparison and len(st.session_state.comparison_reviews) < 5:
                    st.session_state.comparison_reviews.append(analysis_record)
                
                # Auto-save
                auto_save_history()
            
            st.success("✅ Analysis Complete!")
            
            # Result card with animation
            if "FAKE" in result:
                color = "#e74c3c"
                icon = "🚨"
                message = "This review appears to be FAKE"
            elif "GENUINE" in result:
                color = "#2ecc71"
                icon = "✅"
                message = "This review appears to be GENUINE"
            else:
                color = "#f39c12"
                icon = "⚠️"
                message = f"Classification: {result}"
            
            animation_class = "slide-in" if st.session_state.show_animations else ""
            
            st.markdown(f"""
            <div class="{animation_class}" style="
                background: linear-gradient(135deg, {color}22 0%, {color}11 100%);
                border: 3px solid {color};
                padding: 30px;
                border-radius: 15px;
                text-align: center;
                margin: 20px 0;
                box-shadow: 0 10px 30px {color}33;
            ">
                <h1 style="color: {color}; margin: 0; font-size: 2.5em;">
                    {icon} {result}
                </h1>
                <p style="color: {color}; margin: 10px 0 0 0; font-size: 1.2em;">
                    {message}
                </p>
                <h2 style="color: {color}; margin: 15px 0 0 0; font-size: 1.8em;">
                    {confidence*100:.2f}% Confidence
                </h2>
            </div>
            """, unsafe_allow_html=True)
            
            if show_detailed_analysis:
                # Confidence gauge and probability chart side by side
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🎯 Confidence Gauge")
                    st.plotly_chart(create_gauge_chart(confidence, result), use_container_width=True)
                
                with col2:
                    st.markdown("### 📊 Probability Distribution")
                    
                    if probabilities is not None:
                        prob_dict = {LABEL_MAP[i]: probabilities[i].item() for i in range(len(probabilities))}
                        
                        labels = list(prob_dict.keys())
                        values = [prob_dict[label] * 100 for label in labels]
                        colors_list = []
                        
                        for label in labels:
                            if "FAKE" in label:
                                colors_list.append('#e74c3c')
                            elif "GENUINE" in label:
                                colors_list.append('#2ecc71')
                            else:
                                colors_list.append('#f39c12')
                        
                        fig = go.Figure(data=[
                            go.Bar(
                                x=labels,
                                y=values,
                                marker_color=colors_list,
                                text=[f'{v:.1f}%' for v in values],
                                textposition='auto',
                            )
                        ])
                        
                        fig.update_layout(
                            height=300,
                            margin=dict(t=20, b=20, l=20, r=20),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color=colors['text_primary'], size=14, family="Poppins"),
                            yaxis=dict(
                                title="Confidence (%)",
                                gridcolor='rgba(128,128,128,0.2)'
                            ),
                            xaxis=dict(
                                title="Classification"
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            
            # Review details
            with st.expander("📝 Review Details"):
                st.markdown(f"**Review Text:** {review_text}")
                st.markdown(f"**Word Count:** {len(review_text.split())} words")
                st.markdown(f"**Character Count:** {len(review_text)} characters")
                st.markdown(f"**Analyzed At:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                st.markdown(f"**Confidence Threshold:** {st.session_state.confidence_threshold * 100:.0f}%")
                
                if add_to_comparison:
                    st.success("✅ Added to comparison list")
        else:
            st.warning("⚠️ Please enter a review to analyze.")

# ============================================================================
# PAGE 2: BATCH ANALYSIS (PRESERVED WITH ENHANCEMENTS)
# ============================================================================

elif app_mode == "📊 Batch Analysis":
    st.markdown("## 📊 Batch Review Analysis")
    st.markdown("Analyze multiple reviews at once. Enter one review per line.")
    
    batch_text = st.text_area(
        "Enter Multiple Reviews (one per line):",
        height=200,
        placeholder="Review 1...\nReview 2...\nReview 3..."
    )
    
    col_batch1, col_batch2 = st.columns(2)
    with col_batch1:
        show_progress = st.checkbox("Show progress animation", value=True)
    with col_batch2:
        add_to_history = st.checkbox("Add to history", value=True)
    
    if st.button("🚀 Analyze Batch", use_container_width=True, type="primary"):
        lines = [line.strip() for line in batch_text.split('\n') if line.strip()]
        
        if lines:
            st.info(f"📊 Analyzing {len(lines)} reviews...")
            
            results = []
            progress_bar = st.progress(0) if show_progress else None
            status_text = st.empty() if show_progress else None
            
            for i, line in enumerate(lines):
                if show_progress:
                    status_text.text(f"Processing review {i+1}/{len(lines)}...")
                result, confidence, _ = analyze_review(line)
                
                record = {
                    'text': line,
                    'result': result,
                    'confidence': confidence,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                results.append(record)
                
                if add_to_history:
                    st.session_state.history.append(record)
                    st.session_state.stats['total'] += 1
                    if "FAKE" in result:
                        st.session_state.stats['fake'] += 1
                    elif "GENUINE" in result:
                        st.session_state.stats['genuine'] += 1
                
                if show_progress:
                    progress_bar.progress((i + 1) / len(lines))
                    time.sleep(0.1)
            
            if show_progress:
                status_text.empty()
                progress_bar.empty()
            
            st.success(f"✅ Analyzed {len(results)} reviews!")
            
            # Summary statistics
            fake_count = sum(1 for r in results if "FAKE" in r['result'])
            genuine_count = sum(1 for r in results if "GENUINE" in r['result'])
            avg_confidence = sum(r['confidence'] for r in results) / len(results)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Reviews", len(results))
            with col2:
                st.metric("Fake Reviews", fake_count, delta=f"{fake_count/len(results)*100:.1f}%")
            with col3:
                st.metric("Genuine Reviews", genuine_count, delta=f"{genuine_count/len(results)*100:.1f}%")
            with col4:
                st.metric("Avg Confidence", f"{avg_confidence*100:.1f}%")
            
            # Pie chart of results
            st.markdown("### 📊 Results Distribution")
            fig = go.Figure(data=[go.Pie(
                labels=['Fake', 'Genuine'],
                values=[fake_count, genuine_count],
                marker=dict(colors=['#e74c3c', '#2ecc71']),
                hole=0.4
            )])
            
            fig.update_layout(
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': colors['text_primary'], 'family': "Poppins"}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Export batch results
            st.markdown("### 📥 Export Results")
            col_exp1, col_exp2 = st.columns(2)
            
            with col_exp1:
                df = pd.DataFrame(results)
                st.download_button(
                    label="📥 Download as CSV",
                    data=df.to_csv(index=False),
                    file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    type="primary"
                )
            
            with col_exp2:
                st.download_button(
                    label="📥 Download as JSON",
                    data=json.dumps(results, indent=2),
                    file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True,
                    type="primary"
                )
            
            # Display results
            st.markdown("### 📋 Detailed Results")
            
            for idx, result in enumerate(results, 1):
                result_color = "#e74c3c" if "FAKE" in result['result'] else "#2ecc71"
                icon = "🚨" if "FAKE" in result['result'] else "✅"
                
                st.markdown(f"""
                <div style="
                    border-left: 5px solid {result_color};
                    background: {colors['card_bg']};
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 5px;
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: bold; color: {result_color};">
                            {icon} Review #{idx}: {result['result']}
                        </span>
                        <span style="color: {result_color};">
                            {result['confidence']*100:.1f}% confidence
                        </span>
                    </div>
                    <p style="margin: 10px 0 0 0; color: {colors['text_secondary']};">
                        {result['text'][:150]}{'...' if len(result['text']) > 150 else ''}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter at least one review to analyze.")

# ============================================================================
# PAGE 3: COMPARE REVIEWS (NEW FEATURE)
# ============================================================================

elif app_mode == "🔬 Compare Reviews":
    st.markdown("## 🔬 Compare Multiple Reviews")
    st.markdown("Compare up to 5 reviews side-by-side to see patterns and differences.")
    
    if len(st.session_state.comparison_reviews) == 0:
        st.info("💡 No reviews in comparison list. Add reviews from Single Analysis page or manually add below.")
    
    # Manual input for comparison
    st.markdown("### ➕ Add Reviews to Compare")
    
    num_reviews = st.slider("Number of reviews to compare:", 2, 5, min(3, len(st.session_state.comparison_reviews) + 1))
    
    comparison_texts = []
    for i in range(num_reviews):
        review = st.text_area(
            f"Review {i+1}:",
            height=100,
            placeholder=f"Enter review {i+1}...",
            key=f"compare_review_{i}"
        )
        if review.strip():
            comparison_texts.append(review)
    
    if st.button("🔬 Compare All Reviews", use_container_width=True, type="primary"):
        if len(comparison_texts) >= 2:
            with st.spinner("🤖 Analyzing all reviews..."):
                comparison_results = []
                for text in comparison_texts:
                    result, confidence, probabilities = analyze_review(text)
                    comparison_results.append({
                        'text': text,
                        'result': result,
                        'confidence': confidence,
                        'probabilities': probabilities
                    })
            
            st.success("✅ Comparison Complete!")
            
            # Display comparison cards
            st.markdown("### 📊 Comparison Results")
            
            cols = st.columns(len(comparison_results))
            
            for idx, (col, comp) in enumerate(zip(cols, comparison_results), 1):
                with col:
                    result_color = "#e74c3c" if "FAKE" in comp['result'] else "#2ecc71"
                    icon = "🚨" if "FAKE" in comp['result'] else "✅"
                    
                    st.markdown(f"""
                    <div class="comparison-card">
                        <h3 style="color: {result_color}; text-align: center;">
                            {icon} Review {idx}
                        </h3>
                        <div style="text-align: center; margin: 20px 0;">
                            <h2 style="color: {result_color};">{comp['result']}</h2>
                            <h3>{comp['confidence']*100:.1f}%</h3>
                        </div>
                        <p style="font-size: 0.9em; color: {colors['text_secondary']};">
                            {comp['text'][:100]}{'...' if len(comp['text']) > 100 else ''}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Comparison chart
            st.markdown("### 📊 Confidence Comparison")
            
            labels = [f"Review {i+1}" for i in range(len(comparison_results))]
            confidences = [comp['confidence'] * 100 for comp in comparison_results]
            bar_colors = ['#e74c3c' if "FAKE" in comp['result'] else '#2ecc71' for comp in comparison_results]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=labels,
                    y=confidences,
                    marker_color=bar_colors,
                    text=[f"{c:.1f}%" for c in confidences],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                height=400,
                yaxis_title="Confidence (%)",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': colors['text_primary'], 'family': "Poppins"},
                yaxis=dict(gridcolor='rgba(128,128,128,0.2)')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Key insights
            st.markdown("### 🔍 Key Insights")
            
            fake_count = sum(1 for c in comparison_results if "FAKE" in c['result'])
            genuine_count = sum(1 for c in comparison_results if "GENUINE" in c['result'])
            avg_confidence = sum(c['confidence'] for c in comparison_results) / len(comparison_results)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Fake Reviews", fake_count)
            with col2:
                st.metric("Genuine Reviews", genuine_count)
            with col3:
                st.metric("Average Confidence", f"{avg_confidence*100:.1f}%")
            
        else:
            st.warning("⚠️ Please enter at least 2 reviews to compare.")
    
    # Show saved comparison reviews
    if st.session_state.comparison_reviews:
        st.markdown("---")
        st.markdown("### 💾 Saved Comparison Reviews")
        
        for idx, comp in enumerate(st.session_state.comparison_reviews, 1):
            result_color = "#e74c3c" if "FAKE" in comp['result'] else "#2ecc71"
            icon = "🚨" if "FAKE" in comp['result'] else "✅"
            
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"""
                <span class="badge badge-{('fake' if 'FAKE' in comp['result'] else 'genuine')}">
                    {icon} {comp['result']} - {comp['confidence']*100:.1f}%
                </span>
                <p style="color: {colors['text_secondary']}; font-size: 0.9em; margin: 5px 0;">
                    {comp['text'][:100]}{'...' if len(comp['text']) > 100 else ''}
                </p>
                """, unsafe_allow_html=True)
            with col2:
                if st.button("🗑️", key=f"remove_comp_{idx}", help="Remove from comparison"):
                    st.session_state.comparison_reviews.pop(idx-1)
                    st.rerun()
        
        if st.button("🗑️ Clear All Comparisons", use_container_width=True):
            st.session_state.comparison_reviews = []
            st.rerun()

# ============================================================================
# PAGE 4: HISTORY (ENHANCED WITH FILTERS)
# ============================================================================

elif app_mode == "📜 History":
    st.markdown("## 📜 Analysis History")
    
    if st.session_state.history:
        st.markdown(f"**Total analyses:** {len(st.session_state.history)}")
        
        # Filter options
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        with col_filter1:
            filter_type = st.selectbox(
                "Filter by Result:",
                ["All", "FAKE", "GENUINE", "Others"]
            )
        with col_filter2:
            sort_by = st.selectbox(
                "Sort by:",
                ["Newest First", "Oldest First", "Highest Confidence", "Lowest Confidence"]
            )
        with col_filter3:
            items_per_page = st.selectbox(
                "Items per page:",
                [10, 25, 50, 100, "All"]
            )
        
        # Filter history
        filtered_history = st.session_state.history.copy()
        
        if filter_type != "All":
            filtered_history = [h for h in filtered_history if filter_type in h['result']]
        
        # Sort history
        if sort_by == "Newest First":
            filtered_history = list(reversed(filtered_history))
        elif sort_by == "Highest Confidence":
            filtered_history = sorted(filtered_history, key=lambda x: x['confidence'], reverse=True)
        elif sort_by == "Lowest Confidence":
            filtered_history = sorted(filtered_history, key=lambda x: x['confidence'])
        
        # Pagination
        if items_per_page != "All":
            page_size = int(items_per_page)
            total_pages = (len(filtered_history) + page_size - 1) // page_size
            
            if 'current_page' not in st.session_state:
                st.session_state.current_page = 1
            
            col_page1, col_page2, col_page3 = st.columns([1, 2, 1])
            with col_page1:
                if st.button("⬅️ Previous", disabled=st.session_state.current_page == 1):
                    st.session_state.current_page -= 1
                    st.rerun()
            with col_page2:
                st.markdown(f"<p style='text-align: center;'>Page {st.session_state.current_page} of {total_pages}</p>", unsafe_allow_html=True)
            with col_page3:
                if st.button("Next ➡️", disabled=st.session_state.current_page >= total_pages):
                    st.session_state.current_page += 1
                    st.rerun()
            
            start_idx = (st.session_state.current_page - 1) * page_size
            end_idx = min(start_idx + page_size, len(filtered_history))
            filtered_history = filtered_history[start_idx:end_idx]
        
        # Display history
        for idx, item in enumerate(filtered_history, 1):
            result_color = "#e74c3c" if "FAKE" in item['result'] else "#2ecc71"
            icon = "🚨" if "FAKE" in item['result'] else "✅"
            
            with st.expander(f"{icon} {item['result']} - {item['timestamp']} - {item['confidence']*100:.1f}%"):
                st.markdown(f"**Result:** {item['result']}")
                st.markdown(f"**Confidence:** {item['confidence']*100:.2f}%")
                st.markdown(f"**Timestamp:** {item['timestamp']}")
                st.markdown(f"**Review:**")
                st.code(item['text'])
                
                col_action1, col_action2 = st.columns(2)
                with col_action1:
                    if st.button("🔬 Add to Comparison", key=f"add_comp_hist_{idx}"):
                        if len(st.session_state.comparison_reviews) < 5:
                            st.session_state.comparison_reviews.append(item)
                            st.success("Added to comparison!")
                        else:
                            st.warning("Comparison list is full (max 5)")
                with col_action2:
                    if st.button("🔍 Re-analyze", key=f"reanalyze_hist_{idx}"):
                        result, confidence, _ = analyze_review(item['text'])
                        st.info(f"New result: {result} ({confidence*100:.1f}%)")
    else:
        st.info("📭 No analysis history yet. Start by analyzing some reviews!")

# ============================================================================
# PAGE 5: ANALYTICS (NEW FEATURE)
# ============================================================================

elif app_mode == "📈 Analytics":
    st.markdown("## 📈 Analytics Dashboard")
    
    if st.session_state.history:
        st.markdown("### 📊 Overview Statistics")
        
        # Summary metrics
        total = len(st.session_state.history)
        fake_count = sum(1 for h in st.session_state.history if "FAKE" in h['result'])
        genuine_count = sum(1 for h in st.session_state.history if "GENUINE" in h['result'])
        avg_confidence = sum(h['confidence'] for h in st.session_state.history) / total
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Analyzed", total)
        with col2:
            st.metric("Fake Rate", f"{(fake_count/total)*100:.1f}%", 
                     delta=f"{fake_count} reviews", delta_color="inverse")
        with col3:
            st.metric("Genuine Rate", f"{(genuine_count/total)*100:.1f}%",
                     delta=f"{genuine_count} reviews")
        with col4:
            st.metric("Avg Confidence", f"{avg_confidence*100:.1f}%")
        
        # Timeline chart
        st.markdown("### 📈 Analysis Trend Over Time")
        timeline_fig = create_history_chart()
        if timeline_fig:
            st.plotly_chart(timeline_fig, use_container_width=True)
        
        # Confidence distribution
        st.markdown("### 📊 Confidence Distribution")
        
        confidences = [h['confidence'] * 100 for h in st.session_state.history]
        
        fig = go.Figure(data=[go.Histogram(
            x=confidences,
            nbinsx=20,
            marker_color='#FF4B4B'
        )])
        
        fig.update_layout(
            xaxis_title="Confidence (%)",
            yaxis_title="Count",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': colors['text_primary'], 'family': "Poppins"},
            xaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
            yaxis=dict(gridcolor='rgba(128,128,128,0.2)')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed breakdown
        st.markdown("### 🔍 Detailed Breakdown")
        
        df = pd.DataFrame(st.session_state.history)
        
        col_table1, col_table2 = st.columns(2)
        
        with col_table1:
            st.markdown("#### By Result Type")
            result_counts = df['result'].value_counts()
            st.dataframe(result_counts, use_container_width=True)
        
        with col_table2:
            st.markdown("#### Confidence Ranges")
            df['confidence_range'] = pd.cut(df['confidence'], 
                                           bins=[0, 0.5, 0.7, 0.9, 1.0],
                                           labels=['0-50%', '50-70%', '70-90%', '90-100%'])
            range_counts = df['confidence_range'].value_counts().sort_index()
            st.dataframe(range_counts, use_container_width=True)
        
    else:
        st.info("📭 No data available yet. Start analyzing reviews to see analytics!")

# ============================================================================
# PAGE 6: MODEL INFO (PRESERVED)
# ============================================================================

elif app_mode == "🤖 Model Info":
    st.markdown("## 🤖 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="feature-card">
            <h3>🧠 Model Architecture</h3>
            <p><strong>Base Model:</strong> DistilBERT</p>
            <p><strong>Parameters:</strong> ~66M</p>
            <p><strong>Framework:</strong> PyTorch</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="feature-card">
            <h3>🎯 Classification Info</h3>
            <p><strong>Number of Labels:</strong> {num_labels}</p>
            <p><strong>Label Mapping:</strong></p>
            <ul>
                {"".join([f"<li>{k}: {v}</li>" for k, v in LABEL_MAP.items()])}
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="feature-card">
            <h3>⚙️ Technical Details</h3>
            <p><strong>Max Sequence Length:</strong> 512 tokens</p>
            <p><strong>Device:</strong> CPU</p>
            <p><strong>Loaded At:</strong> {st.session_state.model_info.get('loaded_at', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Performance</h3>
            <p>The model uses contextualized word embeddings to understand review patterns.</p>
            <p><strong>Features Analyzed:</strong></p>
            <ul>
                <li>Sentiment patterns</li>
                <li>Language authenticity</li>
                <li>Review structure</li>
                <li>Contextual coherence</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# PAGE 7: SETTINGS (NEW FEATURE)
# ============================================================================

elif app_mode == "⚙️ Settings":
    st.markdown("## ⚙️ Application Settings")
    
    st.markdown("### 🎯 Detection Settings")
    
    new_threshold = st.slider(
        "Confidence Threshold:",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.confidence_threshold,
        step=0.05,
        help="Set the minimum confidence threshold for classification"
    )
    
    if new_threshold != st.session_state.confidence_threshold:
        st.session_state.confidence_threshold = new_threshold
        st.success(f"✅ Threshold updated to {new_threshold*100:.0f}%")
    
    st.markdown("---")
    
    st.markdown("### 🎨 Display Settings")
    
    col_set1, col_set2 = st.columns(2)
    
    with col_set1:
        st.session_state.show_animations = st.checkbox(
            "🎬 Enable animations",
            value=st.session_state.show_animations,
            help="Toggle animations for a smoother experience"
        )
    
    with col_set2:
        st.session_state.auto_save_history = st.checkbox(
            "💾 Auto-save history",
            value=st.session_state.auto_save_history,
            help="Automatically save history to JSON file"
        )
    
    st.markdown("---")
    
    st.markdown("### 🎨 Theme")
    
    col_theme1, col_theme2 = st.columns(2)
    
    with col_theme1:
        if st.button("🌙 Dark Mode", use_container_width=True, 
                    disabled=st.session_state.theme == 'dark'):
            st.session_state.theme = 'dark'
            st.rerun()
    
    with col_theme2:
        if st.button("☀️ Light Mode", use_container_width=True,
                    disabled=st.session_state.theme == 'light'):
            st.session_state.theme = 'light'
            st.rerun()
    
    st.markdown("---")
    
    st.markdown("### 🗑️ Data Management")
    
    col_data1, col_data2 = st.columns(2)
    
    with col_data1:
        if st.button("🗑️ Clear All History", use_container_width=True, type="secondary"):
            if st.button("⚠️ Confirm Clear History", use_container_width=True):
                st.session_state.history = []
                st.success("✅ History cleared!")
                st.rerun()
    
    with col_data2:
        if st.button("🔄 Reset All Statistics", use_container_width=True, type="secondary"):
            if st.button("⚠️ Confirm Reset Stats", use_container_width=True):
                st.session_state.stats = {'total': 0, 'fake': 0, 'genuine': 0}
                st.success("✅ Statistics reset!")
                st.rerun()
    
    st.markdown("---")
    
    st.markdown("### 📥 Import/Export")
    
    # Export all data
    if st.button("📥 Export All Data", use_container_width=True, type="primary"):
        export_data = {
            'history': st.session_state.history,
            'stats': st.session_state.stats,
            'settings': {
                'threshold': st.session_state.confidence_threshold,
                'theme': st.session_state.theme,
                'show_animations': st.session_state.show_animations,
                'auto_save': st.session_state.auto_save_history
            },
            'export_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        st.download_button(
            label="⬇️ Download Complete Backup",
            data=json.dumps(export_data, indent=2),
            file_name=f"review_guardian_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    # Import data
    uploaded_file = st.file_uploader("📤 Import Backup File", type=['json'])
    
    if uploaded_file is not None:
        try:
            import_data = json.load(uploaded_file)
            
            if st.button("⚠️ Confirm Import (will overwrite current data)", use_container_width=True):
                if 'history' in import_data:
                    st.session_state.history = import_data['history']
                if 'stats' in import_data:
                    st.session_state.stats = import_data['stats']
                if 'settings' in import_data:
                    settings = import_data['settings']
                    st.session_state.confidence_threshold = settings.get('threshold', 0.5)
                    st.session_state.theme = settings.get('theme', 'dark')
                    st.session_state.show_animations = settings.get('show_animations', True)
                    st.session_state.auto_save_history = settings.get('auto_save', True)
                
                st.success("✅ Data imported successfully!")
                st.rerun()
        except Exception as e:
            st.error(f"❌ Failed to import data: {str(e)}")
    
    st.markdown("---")
    
    st.markdown("### ℹ️ About")
    
    st.markdown(f"""
    <div class="feature-card">
        <h4>🛡️ AI Review Guardian</h4>
        <p><strong>Version:</strong> 2.0 Enhanced</p>
        <p><strong>Model:</strong> DistilBERT</p>
        <p><strong>Current Theme:</strong> {st.session_state.theme.title()}</p>
        <p><strong>Total Analyses:</strong> {len(st.session_state.history)}</p>
        <p><strong>Session Started:</strong> {st.session_state.model_info.get('loaded_at', 'N/A')}</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: {colors['text_secondary']}; padding: 20px;">
    <p>🛡️ AI Review Guardian v2.0 | Powered by DistilBERT & PyTorch</p>
    <p style="font-size: 0.9em;">Advanced Neural Network for Review Authenticity Detection</p>
    <p style="font-size: 0.8em; margin-top: 10px;">
        Current Theme: {st.session_state.theme.title()} | 
        Analyses: {len(st.session_state.history)} | 
        Threshold: {st.session_state.confidence_threshold*100:.0f}%
    </p>
</div>
""", unsafe_allow_html=True)
