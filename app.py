import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import streamlit as st
import time
import requests
from streamlit_lottie import st_lottie

# 1. Setup Page Config
st.set_page_config(page_title="My Cool App", layout="wide")

# 2. Function to load Lottie Animations
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# 3. The Logic: Only run this on the first load
if 'first_load' not in st.session_state:
    st.session_state['first_load'] = True

if st.session_state['first_load']:
    # Create an empty container to hold the animation
    loader_placeholder = st.empty()
    
    # Load a cool animation (this URL is a tech/network animation)
    lottie_url = "https://assets9.lottiefiles.com/packages/lf20_p8bfn5to.json"
    lottie_json = load_lottieurl(lottie_url)

    # Render the animation in the container
    with loader_placeholder.container():
        # You can use columns to center the animation if needed
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st_lottie(lottie_json, height=300, key="loader")
            st.markdown("<h3 style='text-align: center;'>Setting things up...</h3>", unsafe_allow_html=True)
    
    # Simulate loading delay (or put your heavy computation here)
    time.sleep(3) 
    
    # Clear the placeholder (making the animation disappear)
    loader_placeholder.empty()
    
    # Mark first load as false so it doesn't happen on button clicks
    st.session_state['first_load'] = False

# --- YOUR MAIN APP CODE STARTS HERE ---
st.title("Welcome to the Dashboard")
st.write("The animation is gone, and the app is ready!")
st.button("Click me (I won't trigger the animation again)")

# --- SAFELY IMPORT PLOTLY ---
# This block prevents the app from crashing if plotly is missing
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("⚠️ Plotly is not installed. Graphs will be disabled.")

# --- CONFIGURATION ---
LOOKBACK = 10  # The model needs 10 steps of history
FEATURES_TO_USE = ['Rolling_RPS', 'Packet_Variance', 'Bytes_Sent', 'Bytes_Received']

# --- 1. LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    try:
        model = tf.keras.models.load_model('agent_hunter_model.h5')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model/scaler: {e}")
        return None, None

model, scaler = load_resources()

# --- 2. FEATURE ENGINEERING FUNCTION ---
def engineer_features(df):
    """
    Takes raw logs (Timestamp, IP, Packet Length) and calculates
    Behavioral Features (RPS, Variance).
    """
    st.info("Running Feature Engineering on Raw Logs...")
    
    # Ensure Timestamp is datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Sort for rolling window calculations
    df = df.sort_values(by=['Source IP Address', 'Timestamp'])
    
    # Set index to Timestamp for rolling calcs, but keep it as a column too
    df = df.set_index('Timestamp', drop=False)
    
    # --- A. CALCULATE ROLLING RPS (Velocity) ---
    # Count packets per second per IP
    df['Rolling_RPS'] = df.groupby('Source IP Address')['Source IP Address'].rolling('1s').count().values
    
    # --- B. CALCULATE PACKET VARIANCE (I/O Asymmetry) ---
    # Variance of packet length over the last 1 second
    df['Packet_Variance'] = df.groupby('Source IP Address')['Packet Length'].rolling('1s').var().values
    
    # --- C. BYTES SENT/RECEIVED ---
    # For this POC, we treat Packet Length as Bytes Sent (Simplification)
    df['Bytes_Sent'] = df['Packet Length']
    df['Bytes_Received'] = df['Packet Length'] # Placeholder if full bidirectional data isn't available
    
    # Clean up NaNs (first row of any window is usually NaN)
    df = df.fillna(0)
    
    # Reset index so Timestamp is just a column again
    df = df.reset_index(drop=True)
    
    return df

# --- 3. MAIN APP UI ---
st.title("🛡️ Agent Hunter v2: AI Behavioral Detection")
st.markdown("Upload server logs to detect **GTG-1002** agentic behavior.")

uploaded_file = st.file_uploader("Upload CSV Logs", type=['csv'])

if uploaded_file is not None and model is not None:
    # Load Raw Data
    raw_df = pd.read_csv(uploaded_file)
    st.write("### 1. Raw Data Preview")
    st.dataframe(raw_df.head())

    # Check if we need to engineer features
    if 'Packet Length' in raw_df.columns and 'Rolling_RPS' not in raw_df.columns:
        df_features = engineer_features(raw_df.copy())
    else:
        df_features = raw_df.copy() # Assume features already exist
        
    # Scale Data
    try:
        scaled_features = scaler.transform(df_features[FEATURES_TO_USE])
        # Create a temporary dataframe for easy slicing later
        df_scaled = pd.DataFrame(scaled_features, columns=FEATURES_TO_USE)
        df_scaled['Source IP Address'] = df_features['Source IP Address'].values
        df_scaled['Timestamp'] = df_features['Timestamp'].values
        df_scaled['Original_Index'] = df_features.index.values # Track IDs!
    except KeyError as e:
        st.error(f"Missing Columns for analysis: {e}")
        st.stop()

    # --- 4. SEQUENCE GENERATION (THE FIX) ---
    st.write("### 2. Generating Sequences...")
    
    X_sequences = []
    valid_indices = [] # The IDs of the rows we actually predict
    
    # Group by IP to ensure we don't mix traffic from different users
    for ip, group in df_scaled.groupby('Source IP Address'):
        group_values = group[FEATURES_TO_USE].values
        original_idxs = group['Original_Index'].values
        
        if len(group) > LOOKBACK:
            for i in range(LOOKBACK, len(group)):
                # Create window
                X_sequences.append(group_values[i-LOOKBACK:i])
                # Store the ID of the target row
                valid_indices.append(original_idxs[i])
    
    X_sequences = np.array(X_sequences)
    
    if len(X_sequences) > 0:
        st.success(f"Generated {len(X_sequences)} valid sequences for analysis.")
        
        # --- 5. PREDICTION & ALIGNMENT ---
        predictions = model.predict(X_sequences)
        
        # Filter the original feature DF to match ONLY the predicted rows
        results_df = df_features.loc[valid_indices].copy()
        
        # Now we can safely assign the column
        results_df['AI_Probability'] = predictions
        results_df['Threat_Label'] = results_df['AI_Probability'].apply(lambda x: '🚨 AGENT' if x > 0.5 else '✅ HUMAN')
        
        # --- 6. DISPLAY RESULTS ---
        st.write("### 3. Detection Results")
        
        threats = results_df[results_df['Threat_Label'] == '🚨 AGENT']
        if not threats.empty:
            st.error(f"⚠️ ALERT: Detected {len(threats)} malicious packets!")
        else:
            st.success("System Clean. Normal traffic patterns.")
            
        st.dataframe(results_df[['Timestamp', 'Source IP Address', 'Rolling_RPS', 'Packet_Variance', 'AI_Probability', 'Threat_Label']])
        
        # Visualization
        if PLOTLY_AVAILABLE:
            try:
                fig = px.scatter(
                    results_df, 
                    x='Rolling_RPS', 
                    y='Packet_Variance', 
                    color='Threat_Label',
                    title="Visual Proof: Agent vs Human Traffic",
                    color_discrete_map={'🚨 AGENT': 'red', '✅ HUMAN': 'blue'},
                    hover_data=['Source IP Address']
                )
                st.plotly_chart(fig)
            except Exception as e:
                st.warning(f"Could not render plot: {e}")
        else:
            st.warning("⚠️ Plotly library not found. Install it with `pip install plotly` to see the visualization.")
            
    else:

        st.warning(f"Not enough data. Each IP needs at least {LOOKBACK + 1} packets.")
