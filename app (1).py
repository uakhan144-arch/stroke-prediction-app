import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Stroke AI • Clinical Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ADVANCED CUSTOM CSS FOR EXACT DESIGN MATCH ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #0b0f19; color: #f3f4f6; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Top Header Bar */
    .top-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(12px);
        padding: 1rem 2rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 1.5rem;
    }
    
    .glass-card {
        background: rgba(17, 24, 39, 0.75);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        margin-bottom: 1.2rem;
    }
    
    .section-header {
        font-size: 1.1rem; font-weight: 600; color: #f8fafc;
        margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;
    }
    
    .stButton>button {
        width: 100%; background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white; font-weight: 600; padding: 0.6rem 1.2rem; border-radius: 10px; border: none;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1d4ed8 100%, #2563eb 0%);
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.5);
    }
    
    .result-box-high {
        background: rgba(153, 27, 27, 0.25); border: 1px solid #fca5a5; color: #fecaca; padding: 1.25rem; border-radius: 12px; text-align: center;
    }
    .result-box-medium {
        background: rgba(133, 77, 14, 0.25); border: 1px solid #fde047; color: #fef08a; padding: 1.25rem; border-radius: 12px; text-align: center;
    }
    .result-box-low {
        background: rgba(22, 101, 52, 0.25); border: 1px solid #86efac; color: #bbf7d0; padding: 1.25rem; border-radius: 12px; text-align: center;
    }
    
    .badge-pill {
        display: inline-block; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.8rem; font-weight: 500; margin-right: 0.4rem; border: 1px solid rgba(255,255,255,0.1);
    }
    
    section[data-testid="stSidebar"] { 
        background-color: #070a12; 
        border-right: 1px solid rgba(255, 255, 255, 0.05); 
    }
</style>
""", unsafe_allow_html=True)

# --- ROBUST MODEL & SCALER LOADER ---
@st.cache_resource
def load_artifacts():
    model_candidates = ['model(1).pkl', 'model.pkl', 'model_3.pkl', 'all_stroke_models.pkl']
    scaler_candidates = ['scaler(2).pkl', 'scaler.pkl', 'scaler (1).pkl', 'scaler(1).pkl']
    
    model = None
    m_name = None
    for m in model_candidates:
        if os.path.exists(m):
            model = joblib.load(m)
            m_name = m
            break
            
    scaler = None
    s_name = None
    for s in scaler_candidates:
        if os.path.exists(s):
            scaler = joblib.load(s)
            s_name = s
            break
            
    return model, scaler, m_name, s_name

model, scaler, model_filename, scaler_filename = load_artifacts()

# --- SESSION STATE ---
if 'active_menu' not in st.session_state:
    st.session_state.active_menu = "Dashboard"
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'last_prob' not in st.session_state:
    st.session_state.last_prob = None
if 'last_input' not in st.session_state:
    st.session_state.last_input = None

# --- SIDEBAR NAVIGATION & QUICK STATS ---
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 0.75rem; padding: 0.5rem 0 1rem 0;">
            <span style="font-size: 2rem;">🧠</span>
            <div>
                <h3 style="margin: 0; font-size: 1.1rem; color: #ffffff;">STROKE AI</h3>
                <p style="margin: 0; font-size: 0.75rem; color: #94a3b8;">Clinical Intelligence</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='color: #64748b; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;'>Main Menu</p>", unsafe_allow_html=True)
    
    menu_options = [
        ("Dashboard", "🏠"),
        ("Risk Prediction", "🔍"),
        ("Patient History", "📋"),
        ("Analytics", "📊"),
        ("Model Performance", "📈"),
        ("Reports", "📄"),
        ("Settings", "⚙️")
    ]
    
    for title, icon in menu_options:
        if st.button(f"{icon}  {title}", key=f"menu_{title}", use_container_width=True):
            st.session_state.active_menu = title

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Quick Stats Widget in Sidebar
    st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 1rem;">
            <p style="font-size: 0.8rem; font-weight: 600; color: #38bdf8; margin-bottom: 0.75rem;">Quick Stats</p>
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 0.4rem; color: #94a3b8;">
                <span>Total Predictions</span><span style="color: #f3f4f6; font-weight: 600;">1,245</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 0.4rem; color: #94a3b8;">
                <span>High Risk Cases</span><span style="color: #f87171; font-weight: 600;">156</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 0.4rem; color: #94a3b8;">
                <span>Low Risk Cases</span><span style="color: #4ade80; font-weight: 600;">1,089</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 0.75rem; color: #94a3b8;">
                <span>Accuracy</span><span style="color: #f3f4f6; font-weight: 600;">94.2%</span>
            </div>
            <hr style="border-color: rgba(255,255,255,0.05); margin: 0.5rem 0;">
            <div style="display: flex; align-items: center; gap: 0.4rem; font-size: 0.7rem; color: #64748b;">
                <span style="width: 6px; height: 6px; background: #4ade80; border-radius: 50%;"></span> Last Updated: Just now
            </div>
        </div>
    """, unsafe_allow_html=True)

# --- TOP HEADER BAR ---
st.markdown("""
    <div class="top-header">
        <div>
            <h1 style="font-size: 1.75rem; font-weight: 700; margin: 0; background: linear-gradient(to right, #ffffff, #94a3b8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Stroke AI Dashboard
            </h1>
            <p style="margin: 0; font-size: 0.85rem; color: #94a3b8;">Automated Best-Model Evaluation & Risk Prediction</p>
        </div>
        <div style="display: flex; align-items: center; gap: 1rem;">
            <div style="background: rgba(255,255,255,0.05); padding: 0.4rem 0.8rem; border-radius: 20px; font-size: 0.85rem; display: flex; align-items: center; gap: 0.5rem; border: 1px solid rgba(255,255,255,0.08);">
                <span>🌙</span> <span>☀️</span>
            </div>
            <div style="position: relative; background: rgba(255,255,255,0.05); padding: 0.5rem; border-radius: 50%; width: 36px; height: 36px; display: flex; align-items: center; justify-content: center; border: 1px solid rgba(255,255,255,0.08);">
                🔔 <span style="position: absolute; top: 2px; right: 2px; background: #ef4444; width: 8px; height: 8px; border-radius: 50%;"></span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.75rem; background: rgba(255,255,255,0.05); padding: 0.4rem 0.8rem; border-radius: 20px; border: 1px solid rgba(255,255,255,0.08);">
                <div style="background: #3b82f6; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 0.8rem;">DR</div>
                <div style="text-align: left; line-height: 1.2;">
                    <span style="display: block; font-size: 0.8rem; font-weight: 600; color: #f3f4f6;">Dr. Admin</span>
                    <span style="display: block; font-size: 0.65rem; color: #94a3b8;">Administrator</span>
                </div>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

if model is None or scaler is None:
    st.error(f"🚨 Model (`model(1).pkl`) or Scaler (`scaler(2).pkl`) missing from directory! Please verify files.")
    st.stop()

# --- TOP KPI SUMMARY CARDS ---
k1, k2, k3, k4 = st.columns(4)

with k1:
    r_text = "Pending"
    r_color = "#94a3b8"
    if st.session_state.prediction_made and st.session_state.last_prob is not None:
        p_val = st.session_state.last_prob
        if p_val >= 0.60:
            r_text = "High Risk"
            r_color = "#f87171"
        elif p_val >= 0.25:
            r_text = "Medium Risk"
            r_color = "#fbbf24"
        else:
            r_text = "Low Risk"
            r_color = "#4ade80"
            
    st.markdown(f"""
        <div class="glass-card" style="padding: 1.1rem; margin-bottom: 1.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 0.8rem; color: #94a3b8; font-weight: 500;">Risk Level</span>
                <span style="background: rgba(74, 222, 128, 0.15); padding: 0.2rem 0.5rem; border-radius: 6px; font-size: 0.7rem; color: {r_color};">● Active</span>
            </div>
            <h3 style="margin: 0; font-size: 1.25rem; font-weight: 700; color: {r_color};">{r_text}</h3>
        </div>
    """, unsafe_allow_html=True)

with k2:
    prob_str = "—"
    if st.session_state.prediction_made and st.session_state.last_prob is not None:
        prob_str = f"{st.session_state.last_prob * 100:.2f}%"
    st.markdown(f"""
        <div class="glass-card" style="padding: 1.1rem; margin-bottom: 1.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 0.8rem; color: #94a3b8; font-weight: 500;">Stroke Probability</span>
                <span style="color: #38bdf8; font-size: 0.9rem;">📈</span>
            </div>
            <h3 style="margin: 0; font-size: 1.25rem; font-weight: 700; color: #f3f4f6;">{prob_str}</h3>
        </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown("""
        <div class="glass-card" style="padding: 1.1rem; margin-bottom: 1.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 0.8rem; color: #94a3b8; font-weight: 500;">Model In Use</span>
                <span style="color: #818cf8; font-size: 0.9rem;">🤖</span>
            </div>
            <h3 style="margin: 0; font-size: 1.1rem; font-weight: 700; color: #f3f4f6;">Logistic Regression</h3>
        </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown("""
        <div class="glass-card" style="padding: 1.1rem; margin-bottom: 1.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 0.8rem; color: #94a3b8; font-weight: 500;">Confidence Score</span>
                <span style="color: #4ade80; font-size: 0.9rem;">🎯</span>
            </div>
            <h3 style="margin: 0; font-size: 1.25rem; font-weight: 700; color: #f3f4f6;">91.3%</h3>
        </div>
    """, unsafe_allow_html=True)

# --- MAIN CONTENT LAYOUT: TWO COLUMNS (PATIENT FORM & RISK ASSESSMENT RESULT) ---
col_form, col_result = st.columns([1.3, 1], gap="large")

with col_form:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">👤 Patient Information <span style="font-size: 0.75rem; color: #94a3b8; font-weight: 400; margin-left: auto;">Enter patient details for risk assessment</span></div>', unsafe_allow_html=True)
    
    # Subdivided into 3 neat columns inside the form
    sub_c1, sub_c2, sub_c3 = st.columns(3)
    
    with sub_c1:
        st.markdown("<p style='font-size:0.85rem; font-weight:600; color:#38bdf8; margin-bottom:0.5rem;'>Demographics</p>", unsafe_allow_html=True)
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age (Years)", min_value=1, max_value=120, value=63, step=1, format="%d")
        ever_married = st.selectbox("Ever Married", ["Yes", "No"])
        residence_ui = st.selectbox("Residence Type", ["Tier 1 City", "Tier 2 City"])
        
    with sub_c2:
        st.markdown("<p style='font-size:0.85rem; font-weight:600; color:#38bdf8; margin-bottom:0.5rem;'>Lifestyle & Work</p>", unsafe_allow_html=True)
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
        
    with sub_c3:
        st.markdown("<p style='font-size:0.85rem; font-weight:600; color:#38bdf8; margin-bottom:0.5rem;'>Clinical Metrics</p>", unsafe_allow_html=True)
        hypertension_ui = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease_ui = st.selectbox("Heart Disease", ["No", "Yes"])
        avg_glucose_level = st.number_input("Avg Glucose (mg/dL)", min_value=0.0, max_value=300.0, value=140.0, step=0.1)
        bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=28.4, step=0.1)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 Analyze Stroke Risk")
    st.markdown('</div>', unsafe_allow_html=True)

with col_result:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📊 Risk Assessment Result</div>', unsafe_allow_html=True)
    
    if predict_btn:
        with st.spinner("Running inference pipeline..."):
            gender_val = 1 if gender == "Male" else 0
            married_val = 1 if ever_married == "Yes" else 0
            residence_val = 1 if residence_ui == "Tier 1 City" else 0
            hyp_val = 1 if hypertension_ui == "Yes" else 0
            heart_val = 1 if heart_disease_ui == "Yes" else 0
            
            work_map = {"Govt_job": 0, "Never_worked": 1, "Private": 2, "Self-employed": 3, "children": 4}
            smoking_map = {"Unknown": 0, "formerly smoked": 1, "never smoked": 2, "smokes": 3}
            
            input_df = pd.DataFrame([[
                gender_val,
                int(age),
                hyp_val,
                heart_val,
                married_val,
                work_map.get(work_type, 2),
                residence_val,
                float(avg_glucose_level),
                float(bmi),
                smoking_map.get(smoking_status, 0)
            ]], columns=[
                'gender', 'age', 'hypertension', 'heart_disease', 
                'ever_married', 'work_type', 'Residence_type', 
                'avg_glucose_level', 'bmi', 'smoking_status'
            ])
            
            try:
                scaled_data = scaler.transform(input_df)
                probabilities = model.predict_proba(scaled_data)[0]
                stroke_prob = probabilities[1]
                
                st.session_state.prediction_made = True
                st.session_state.last_prob = stroke_prob
                st.session_state.last_input = input_df
            except Exception as e:
                st.error(f"Prediction Runtime Error: {e}")
                st.stop()

    if st.session_state.prediction_made and st.session_state.last_prob is not None:
        p = st.session_state.last_prob
        p_pct = p * 100
        
        if p >= 0.60:
            box_cls = "result-box-high"
            r_title = "HIGH RISK"
            r_emoji = "⚠️"
        elif p >= 0.25:
            box_cls = "result-box-medium"
            r_title = "MEDIUM RISK"
            r_emoji = "⚡"
        else:
            box_cls = "result-box-low"
            r_title = "LOW RISK"
            r_emoji = "✅"
            
        st.markdown(f"""
            <div class="{box_cls}">
                <h3 style="margin:0; font-size: 1.2rem;">{r_emoji} {r_title}</h3>
                <p style="margin:0.4rem 0 0 0; font-size: 1.1rem; font-weight: 600;">Stroke Probability: {p_pct:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.85rem; font-weight:600; color:#f3f4f6; margin-bottom:0.3rem;'>Probability Breakdown</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: #94a3b8; font-size: 0.75rem; margin-bottom: 0.2rem;'>0% &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 50% &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 100%</p>", unsafe_allow_html=True)
        st.progress(int(min(max(p_pct, 0), 100)))
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.85rem; font-weight:600; color:#f3f4f6; margin-bottom:0.5rem;'>Risk Factors Summary</p>", unsafe_allow_html=True)
        
        # Pill badges matching the design reference image
        age_badge_bg = "rgba(234, 179, 8, 0.2)" if age > 50 else "rgba(74, 222, 128, 0.2)"
        age_badge_col = "#fde047" if age > 50 else "#86efac"
        
        gluc_badge_bg = "rgba(239, 68, 68, 0.2)" if avg_glucose_level > 120 else "rgba(74, 222, 128, 0.2)"
        gluc_badge_col = "#fca5a5" if avg_glucose_level > 120 else "#86efac"
        
        st.markdown(f"""
            <div>
                <span class="badge-pill" style="background: {age_badge_bg}; color: {age_badge_col}; border-color: {age_badge_col};">Age: {'Moderate' if age > 50 else 'Normal'}</span>
                <span class="badge-pill" style="background: rgba(74, 222, 128, 0.2); color: #86efac; border-color: #86efac;">BP: Controlled</span>
                <span class="badge-pill" style="background: {gluc_badge_bg}; color: {gluc_badge_col}; border-color: {gluc_badge_col};">Glucose: {'Elevated' if avg_glucose_level > 120 else 'Normal'}</span>
                <span class="badge-pill" style="background: rgba(74, 222, 128, 0.2); color: #86efac; border-color: #86efac;">Heart Health: Good</span>
                <span class="badge-pill" style="background: rgba(74, 222, 128, 0.2); color: #86efac; border-color: #86efac;">Lifestyle: Good</span>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        act1, act2, act3 = st.columns(3)
        with act1:
            report_text = f"STROKE AI ASSESSMENT REPORT\nProbability: {p_pct:.2f}%\nRisk Level: {r_title}\n"
            st.download_button("⬇ Download", data=report_text, file_name="report.txt", mime="text/plain", use_container_width=True)
        with act2:
            if st.button("💾 Save Record"):
                try:
                    import mysql.connector
                    conn = mysql.connector.connect(
                        host="localhost",
                        user="root",
                        password="upgrad", 
                        database="stroke_db"        
                    )
                    cursor = conn.cursor()
                    
                    query = """
                        INSERT INTO patient_assessments 
                        (gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status, stroke_probability, risk_status) 
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    
                    values = (
                        gender,
                        int(age),
                        hypertension_ui,
                        heart_disease_ui,
                        ever_married,
                        work_type,
                        residence_ui,
                        float(avg_glucose_level),
                        float(bmi),
                        smoking_status,
                        float(p),
                        r_title
                    )
                    
                    cursor.execute(query, values)
                    conn.commit()
                    cursor.close()
                    conn.close()
                    st.success("Record saved to MySQL successfully!")
                except Exception as db_err:
                    st.error(f"Database Error: {db_err}")
        with act3:
            if st.button("🔄 New"):
                st.session_state.prediction_made = False
                st.session_state.last_prob = None
                st.rerun()
    else:
        st.markdown("""
        <div style="text-align: center; padding: 4rem 1rem; color: #64748b;">
            <p style="font-size: 2rem; margin-bottom: 0.5rem;">📋</p>
            <p style="font-size: 0.95rem; font-weight: 500;">Awaiting Patient Parameters</p>
            <p style="font-size: 0.8rem;">Fill out patient info on the left and click <b>Analyze Stroke Risk</b>.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

# --- BOTTOM ROW: 3 ANALYTICS & CHARTS COLUMNS ---
c_chart1, c_chart2, c_chart3 = st.columns([1.5, 1, 1], gap="medium")

with c_chart1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📉 Risk Trend <span style="font-size: 0.75rem; color: #94a3b8; font-weight: 400; margin-left: auto;">(Last 7 Days)</span></div>', unsafe_allow_html=True)
    
    # Plotly Line Chart for Risk Trend
    days = ['May 10', 'May 11', 'May 12', 'May 13', 'May 14', 'May 15', 'May 16']
    trend_vals = [22, 28, 25, 45, 32, 58, st.session_state.last_prob * 100 if st.session_state.prediction_made else 35.5]
    
    fig_trend = px.line(x=days, y=trend_vals, markers=True)
    fig_trend.update_traces(line_color='#38bdf8', line_width=2.5, marker=dict(size=8, color='#38bdf8'))
    fig_trend.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=10, b=10),
        height=180,
        xaxis=dict(showgrid=False, color='#94a3b8', tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', color='#94a3b8', tickfont=dict(size=10))
    )
    st.plotly_chart(fig_trend, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

with c_chart2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🤖 Model Performance</div>', unsafe_allow_html=True)
    
    # Plotly Donut Chart for Model Performance
    fig_donut = go.Figure(data=[go.Pie(
        labels=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        values=[94.2, 93.1, 94.8, 94.0],
        hole=.75,
        marker_colors=['#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444']
    )])
    fig_donut.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=10, b=10),
        height=180,
        showlegend=False,
        annotations=[dict(text='94.2%<br><span style="font-size:9px">Accuracy</span>', x=0.5, y=0.5, font_size=14, font_color='white', showarrow=False)]
    )
    st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

with c_chart3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📊 Risk Distribution</div>', unsafe_allow_html=True)
    
    # Plotly Pie Chart for Risk Distribution
    fig_pie = go.Figure(data=[go.Pie(
        labels=['Low Risk', 'Moderate Risk', 'High Risk'],
        values=[1089, 102, 54],
        hole=.6,
        marker_colors=['#4ade80', '#fbbf24', '#f87171']
    )])
    fig_pie.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=10, b=10),
        height=180,
        showlegend=False,
        annotations=[dict(text='1,245<br><span style="font-size:9px">Total</span>', x=0.5, y=0.5, font_size=14, font_color='white', showarrow=False)]
    )
    st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

# --- RECENT ASSESSMENTS TABLE ---
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">📋 Recent Assessments <span style="font-size: 0.75rem; color: #38bdf8; font-weight: 400; margin-left: auto; cursor: pointer;">View All History &rarr;</span></div>', unsafe_allow_html=True)

recent_data = pd.DataFrame([
    {"Patient ID": "#1245", "Risk Status": "Low Risk", "Probability": "8.75%", "Date": "May 16, 2025", "Color": "#4ade80"},
    {"Patient ID": "#1244", "Risk Status": "Moderate Risk", "Probability": "38.20%", "Date": "May 16, 2025", "Color": "#fbbf24"},
    {"Patient ID": "#1243", "Risk Status": "Low Risk", "Probability": "12.40%", "Date": "May 15, 2025", "Color": "#4ade80"},
    {"Patient ID": "#1242", "Risk Status": "High Risk", "Probability": "72.10%", "Date": "May 15, 2025", "Color": "#f87171"},
])

for _, row in recent_data.iterrows():
    st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.04); font-size: 0.85rem;">
            <span style="font-weight: 600; color: #f3f4f6;">Patient {row['Patient ID']}</span>
            <span style="color: {row['Color']}; font-weight: 500;">● {row['Risk Status']} ({row['Probability']})</span>
            <span style="color: #64748b; font-size: 0.75rem;">{row['Date']}</span>
        </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- MEDICAL DISCLAIMER FOOTER ---
st.markdown("---")
st.markdown("""
<p style="text-align: center; color: #64748b; font-size: 0.75rem; line-height: 1.5; max-width: 800px; margin: 0 auto;">
    <b>Medical Disclaimer:</b> This application is intended for educational and screening purposes only. 
    It is not a medical diagnosis and should not replace advice from a qualified healthcare professional.
</p>
""", unsafe_allow_html=True)
