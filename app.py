import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pandas as pd
from database import insert_log, get_all_logs

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Dental Malocclusion AI Triage", page_icon="🦷", layout="wide")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    # Load the custom model from the specified training output folder
    model_path = 'best.pt'
    
    if not os.path.exists(model_path):
        st.error(f"❌ **Critical Error**: Custom model weights not found at exactly:")
        st.code(model_path)
        st.info("Please verify your training pipeline saved `best.pt` to this location. We stopped the app to prevent it from defaulting to standard COCO classes (like 'person').")
        st.stop()
        
    try:
        model = YOLO(model_path)
        model_version = 'YOLOv8-Custom-dental-v1'
        return model, model_version
    except Exception as e:
        st.error(f"❌ **Failed to load YOLO model:** {e}")
        st.stop()

model, model_version = load_model()

# --- HELPER FUNCTIONS ---
def analyze_results(results):
    """Analyzes the YOLO results to determine severity and triage recommendation."""
    detections = results[0].boxes
    num_detections = len(detections)
    
    # Get class names
    names_dict = results[0].names
    
    if num_detections == 0:
        confidence_score = 0.0
        severity_level = "Normal"
        triage_recommendation = "No immediate action required. Routine checkup."
        flag_for_review = "0"
        detected_classes = "None"
    else:
        # Get highest confidence score
        confidences = detections.conf.tolist()
        confidence_score = max(confidences)
        
        # Determine severity based on max confidence
        if confidence_score < 0.50:
            severity_level = "Mild"
            triage_recommendation = "Low confidence detection. Schedule non-urgent orthodontic consultation."
        elif confidence_score < 0.75:
            severity_level = "Moderate"
            triage_recommendation = "Moderate malocclusion signs. Orthodontic assessment recommended soon."
        else:
            severity_level = "Severe"
            triage_recommendation = "High confidence malocclusion detected. High priority assessment."
            
        # Determine auto flag
        flag_for_review = "1" if confidence_score > 0.85 else "0"
        
        # Get detected classes as comma-separated string
        class_ids = detections.cls.tolist()
        class_names_set = set([names_dict[int(cls_id)] for cls_id in class_ids])
        detected_classes = ", ".join(class_names_set)
            
    return num_detections, confidence_score, severity_level, triage_recommendation, flag_for_review, detected_classes

# --- MAIN APP UI ---
st.title("🦷 Dental Malocclusion AI Triage System")
st.markdown("Upload a dental image or X-ray to instantly detect malocclusions and generate an initial triage recommendation.")

tab1, tab2 = st.tabs(["🩺 New Triage", "📋 History Logs"])

with tab1:
    st.header("Patient Image Analysis")
    uploaded_file = st.file_uploader("Upload Image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # 1. Read and display the image
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, use_container_width=True)
            
        # 2. Run Inference
        with st.spinner("Analyzing image..."):
            # Convert PIL image to numpy array for YOLO
            img_array = np.array(image)
            results = model(img_array)
            
            # Extract analysis
            num_detections, confidence_score, severity_level, triage_recommendation, auto_flag, detected_classes = analyze_results(results)
            
            # Plot results on the image
            res_plotted = results[0].plot()
            # Convert BGR (OpenCV) back to RGB (Pillow/Streamlit)
            res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
        with col2:
            st.subheader("AI Detection Results")
            st.image(res_plotted_rgb, use_container_width=True)
            
        # 3. Triage Report
        st.markdown("---")
        st.subheader("Triage Report")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Detections", num_detections)
        c2.metric("Max Confidence", f"{confidence_score:.1%}")
        c3.metric("Severity", severity_level)
        c4.metric("Model", model_version)
        
        # Display recommendations with appropriate styling
        if severity_level == "Normal":
            st.success(f"**Recommendation:** {triage_recommendation}")
        elif severity_level == "Mild":
            st.info(f"**Recommendation:** {triage_recommendation}")
        elif severity_level == "Moderate":
            st.warning(f"**Recommendation:** {triage_recommendation}")
        else:
            st.error(f"**Recommendation:** {triage_recommendation}")
            
        st.write(f"**Detected Classes:** {detected_classes}")
            
        # 4. Action & Logging
        st.markdown("### Action")
        # Pre-select based on the auto_flag logic
        should_flag = st.checkbox("Flag this case for review by a senior orthodontist", value=(auto_flag == "1"))
        final_flag = "1" if should_flag else "0"
        
        if st.button("Log Results to Database"):
            insert_log(
                image_filename=uploaded_file.name,
                num_detections=num_detections,
                confidence_score=confidence_score,
                severity_level=severity_level,
                triage_recommendation=triage_recommendation,
                model_version=model_version,
                flag_for_review=final_flag,
                detected_classes=detected_classes
            )
            st.success(f"Successfully logged the results for `{uploaded_file.name}` into the database!")

with tab2:
    st.header("Previous Triage Logs")
    logs = get_all_logs()
    
    if len(logs) > 0:
        # Convert list of dicts to a Pandas DataFrame for a nice display
        df = pd.DataFrame(logs)
        
        # Format the confidence score to a percentage
        df['confidence_score'] = df['confidence_score'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(df, use_container_width=True)
        
        # Add a download button for CSV export
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name='triage_logs_export.csv',
            mime='text/csv',
        )
    else:
        st.info("No logs found in the database. Go to the 'New Triage' tab to analyze an image.")
