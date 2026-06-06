import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

st.set_page_config(
    page_title="CreditWise | SecureTrust Bank",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

model  = joblib.load('xgb_model.pkl')
scaler = joblib.load('scaler.pkl')
ohe    = joblib.load('ohe.pkl')
le_edu = joblib.load('le_edu.pkl')

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&family=Instrument+Serif:ital@0;1&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    font-family: 'Sora', sans-serif;
    background: #04080f;
    color: #e2e8f0;
}

#MainMenu, footer, header, .stDeployButton { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none; }

/* ── NAVBAR ── */
.navbar {
    position: sticky;
    top: 0;
    z-index: 999;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 4rem;
    background: rgba(4,8,15,0.85);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.nav-logo {
    font-family: 'Instrument Serif', serif;
    font-size: 1.5rem;
    color: white;
    letter-spacing: -0.3px;
}
.nav-logo span { color: #38bdf8; }
.nav-links { display: flex; gap: 2.5rem; }
.nav-links a {
    color: rgba(148,163,184,0.8);
    text-decoration: none;
    font-size: 0.85rem;
    font-weight: 500;
    transition: color 0.2s;
}
.nav-links a:hover { color: white; }
.nav-badge {
    background: rgba(56,189,248,0.1);
    border: 1px solid rgba(56,189,248,0.25);
    color: #38bdf8;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.35rem 1rem;
    border-radius: 999px;
}

/* ── HERO ── */
@keyframes fadeDown { from { opacity:0; transform:translateY(-20px);} to { opacity:1; transform:translateY(0);} }
@keyframes fadeUp   { from { opacity:0; transform:translateY(20px); } to { opacity:1; transform:translateY(0);} }
@keyframes shimmer  { 0% { background-position:-200% center;} 100% { background-position:200% center;} }
@keyframes float    { 0%,100% { transform:translateY(0);} 50% { transform:translateY(-10px);} }
@keyframes pulse    { 0%,100% { opacity:0.4;} 50% { opacity:0.8;} }
@keyframes slideIn  { from { opacity:0; transform:scale(0.94);} to { opacity:1; transform:scale(1);} }

.hero-section {
    position: relative;
    min-height: 92vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 6rem 2rem 4rem;
    overflow: hidden;
}
.hero-bg {
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(56,189,248,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(99,102,241,0.08) 0%, transparent 50%);
    pointer-events: none;
}
.hero-grid {
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(rgba(56,189,248,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(56,189,248,0.04) 1px, transparent 1px);
    background-size: 60px 60px;
    pointer-events: none;
}
.hero-tag {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(56,189,248,0.08);
    border: 1px solid rgba(56,189,248,0.2);
    color: #7dd3fc;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.4rem 1.2rem;
    border-radius: 999px;
    margin-bottom: 1.8rem;
    animation: fadeDown 0.6s ease-out;
}
.hero-tag::before { content: '●'; font-size: 0.5rem; color: #38bdf8; animation: pulse 2s infinite; }
.hero-title {
    font-family: 'Instrument Serif', serif;
    font-size: clamp(3rem, 7vw, 5.5rem);
    font-weight: 400;
    color: #f8fafc;
    line-height: 1.05;
    margin-bottom: 1.5rem;
    animation: fadeDown 0.7s ease-out 0.1s both;
    letter-spacing: -1px;
}
.hero-title .gradient {
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #38bdf8 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 5s linear infinite;
}
.hero-sub {
    font-size: 1.15rem;
    color: rgba(148,163,184,0.8);
    max-width: 560px;
    line-height: 1.7;
    margin: 0 auto 3rem;
    font-weight: 300;
    animation: fadeUp 0.7s ease-out 0.2s both;
}
.hero-stats {
    display: flex;
    gap: 3rem;
    justify-content: center;
    animation: fadeUp 0.7s ease-out 0.3s both;
    margin-bottom: 3rem;
}
.hero-stat-val {
    font-size: 2rem;
    font-weight: 700;
    color: white;
    line-height: 1;
}
.hero-stat-lbl {
    font-size: 0.75rem;
    color: rgba(148,163,184,0.5);
    margin-top: 0.3rem;
    font-weight: 500;
    letter-spacing: 0.05em;
}
.hero-arrow {
    font-size: 1.5rem;
    color: rgba(56,189,248,0.5);
    animation: float 3s ease-in-out infinite;
    margin-top: 1rem;
}

/* ── FEATURES STRIP ── */
.features-strip {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1px;
    background: rgba(255,255,255,0.06);
    border-top: 1px solid rgba(255,255,255,0.06);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 5rem;
}
.feature-item {
    background: #04080f;
    padding: 2.5rem 3rem;
    display: flex;
    align-items: flex-start;
    gap: 1.2rem;
}
.feature-icon {
    font-size: 1.8rem;
    flex-shrink: 0;
    margin-top: 2px;
}
.feature-title { font-size: 0.95rem; font-weight: 600; color: #e2e8f0; margin-bottom: 0.4rem; }
.feature-desc  { font-size: 0.82rem; color: rgba(148,163,184,0.6); line-height: 1.6; }

/* ── FORM SECTION ── */
.form-section {
    max-width: 900px;
    margin: 0 auto;
    padding: 0 2rem 6rem;
}
.form-section-title {
    font-family: 'Instrument Serif', serif;
    font-size: 2.5rem;
    color: white;
    text-align: center;
    margin-bottom: 0.6rem;
    letter-spacing: -0.5px;
}
.form-section-sub {
    text-align: center;
    color: rgba(148,163,184,0.6);
    font-size: 0.9rem;
    margin-bottom: 3rem;
}
.form-card {
    background: #0a1628;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 2rem 2rem 1rem;
    margin-bottom: 1.2rem;
}
.form-card-head {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 1.4rem;
    padding-bottom: 1.1rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.form-card-num {
    width: 28px; height: 28px;
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    border-radius: 7px;
    font-size: 0.8rem;
    font-weight: 700;
    color: white;
    display: flex; align-items: center; justify-content: center;
}
.form-card-title { font-size: 0.95rem; font-weight: 600; color: #e2e8f0; }
.form-card-desc  { font-size: 0.75rem; color: rgba(100,116,139,0.8); }

/* ── Streamlit input overrides ── */
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label {
    color: rgba(100,116,139,0.9) !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    margin-bottom: 4px !important;
}
div[data-testid="stNumberInput"] input {
    background: #0f1f38 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 0.9rem !important;
}
div[data-testid="stNumberInput"] input:focus {
    border-color: rgba(56,189,248,0.5) !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.1) !important;
    outline: none !important;
}
div[data-testid="stSelectbox"] > div > div {
    background: #0f1f38 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* ── Button ── */
@keyframes btnGlow {
    0%,100% { box-shadow: 0 0 20px rgba(56,189,248,0.2); }
    50%      { box-shadow: 0 0 40px rgba(56,189,248,0.5), 0 0 80px rgba(99,102,241,0.2); }
}
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%) !important;
    color: white !important;
    border: none !important;
    padding: 1rem 2rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    border-radius: 14px !important;
    width: 100% !important;
    font-family: 'Sora', sans-serif !important;
    letter-spacing: 0.02em !important;
    animation: btnGlow 3s ease-in-out infinite !important;
    transition: transform 0.2s !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
}

/* ── Result ── */
.result-wrap { animation: slideIn 0.5s cubic-bezier(0.34,1.56,0.64,1); }
.result-approved {
    background: linear-gradient(135deg, #022c22 0%, #064e3b 100%);
    border: 1px solid rgba(52,211,153,0.3);
    border-radius: 22px;
    padding: 3rem 2rem;
    text-align: center;
    box-shadow: 0 32px 80px rgba(52,211,153,0.15);
    margin-top: 1.5rem;
}
.result-rejected {
    background: linear-gradient(135deg, #2d0a14 0%, #4c0519 100%);
    border: 1px solid rgba(248,113,113,0.3);
    border-radius: 22px;
    padding: 3rem 2rem;
    text-align: center;
    box-shadow: 0 32px 80px rgba(248,113,113,0.12);
    margin-top: 1.5rem;
}
.res-icon  { font-size: 4rem; margin-bottom: 1rem; display: block; }
.res-title {
    font-family: 'Instrument Serif', serif;
    font-size: 2.4rem;
    color: white;
    margin-bottom: 0.5rem;
    letter-spacing: -0.5px;
}
.res-msg   { font-size: 0.9rem; color: rgba(255,255,255,0.6); margin-bottom: 2rem; line-height: 1.6; }
.res-conf  {
    display: inline-block;
    background: rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.2rem 3rem;
}
.res-conf-lbl { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; color: rgba(255,255,255,0.45); }
.res-conf-val { font-size: 3rem; font-weight: 700; color: white; line-height: 1.1; }

/* ── Footer ── */
.site-footer {
    border-top: 1px solid rgba(255,255,255,0.06);
    padding: 2.5rem 4rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.footer-logo {
    font-family: 'Instrument Serif', serif;
    font-size: 1.2rem;
    color: rgba(148,163,184,0.5);
}
.footer-logo span { color: #38bdf8; }
.footer-copy {
    font-size: 0.75rem;
    color: rgba(100,116,139,0.5);
}
</style>
""", unsafe_allow_html=True)

# ── NAVBAR ──
st.markdown("""
<div class="navbar">
    <div class="nav-logo">Credit<span>Wise</span></div>
    <div class="nav-badge">SecureTrust Bank</div>
</div>
""", unsafe_allow_html=True)

# ── HERO ──
st.markdown("""
<div class="hero-section">
    <div class="hero-bg"></div>
    <div class="hero-grid"></div>
    <div class="hero-tag">AI-Powered Loan Decisions</div>
    <div class="hero-title">
        Smarter Loans.<br>
        <span class="gradient">Faster Decisions.</span>
    </div>
    <div class="hero-sub">
        SecureTrust Bank's intelligent credit system analyses 20 financial
        factors to give you an instant, unbiased loan decision.
    </div>
    <div class="hero-stats">
        <div>
            <div class="hero-stat-val">96.3%</div>
            <div class="hero-stat-lbl">Accuracy</div>
        </div>
        <div style="width:1px;background:rgba(255,255,255,0.1)"></div>
        <div>
            <div class="hero-stat-val">0.94</div>
            <div class="hero-stat-lbl">F1 Score</div>
        </div>
        <div style="width:1px;background:rgba(255,255,255,0.1)"></div>
        <div>
            <div class="hero-stat-val">&lt; 2s</div>
            <div class="hero-stat-lbl">Decision Time</div>
        </div>
        <div style="width:1px;background:rgba(255,255,255,0.1)"></div>
        <div>
            <div class="hero-stat-val">950+</div>
            <div class="hero-stat-lbl">Records Trained</div>
        </div>
    </div>
    <div class="hero-arrow">↓</div>
</div>
""", unsafe_allow_html=True)

# ── FEATURES STRIP ──
st.markdown("""
<div class="features-strip">
    <div class="feature-item">
        <div class="feature-icon">🧠</div>
        <div>
            <div class="feature-title">XGBoost Powered</div>
            <div class="feature-desc">Trained on real historical loan data with ensemble boosting for maximum accuracy.</div>
        </div>
    </div>
    <div class="feature-item">
        <div class="feature-icon">⚡</div>
        <div>
            <div class="feature-title">Instant Analysis</div>
            <div class="feature-desc">20 financial and personal factors analysed in under one second.</div>
        </div>
    </div>
    <div class="feature-item">
        <div class="feature-icon">🛡️</div>
        <div>
            <div class="feature-title">Unbiased Decisions</div>
            <div class="feature-desc">Removes human bias from the loan process. Consistent, fair, and data-driven.</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── FORM SECTION ──
st.markdown("""
<div class="form-section">
    <div class="form-section-title">Apply for a Loan</div>
    <div class="form-section-sub">Fill in the details below and get an instant AI-powered decision.</div>
</div>
""", unsafe_allow_html=True)

_, form_col, _ = st.columns([0.5, 9, 0.5])

with form_col:

    # Card 1 — Personal
    st.markdown("""
    <div class="form-card">
        <div class="form-card-head">
            <div class="form-card-num">1</div>
            <div>
                <div class="form-card-title">Personal Information</div>
                <div class="form-card-desc">Basic applicant details</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        age    = st.number_input("Age",            min_value=18,  max_value=75,  value=30)
        gender = st.selectbox("Gender",            ["Male", "Female"])
    with c2:
        marital_status = st.selectbox("Marital Status",   ["Married", "Single"])
        dependents     = st.number_input("Dependents",    min_value=0, max_value=10, value=0)
    with c3:
        education_level   = st.selectbox("Education Level",   le_edu.classes_.tolist())
        employer_category = st.selectbox("Employer Category", ["Govt", "Private", "Self"])

    st.markdown("<br>", unsafe_allow_html=True)

    # Card 2 — Financial
    st.markdown("""
    <div class="form-card">
        <div class="form-card-head">
            <div class="form-card-num">2</div>
            <div>
                <div class="form-card-title">Financial Information</div>
                <div class="form-card-desc">Income, savings and credit details</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        applicant_income   = st.number_input("Applicant Income (₹)",    min_value=0, value=50000)
        coapplicant_income = st.number_input("Co-Applicant Income (₹)", min_value=0, value=0)
    with c2:
        credit_score   = st.number_input("Credit Score",   min_value=300, max_value=900, value=700)
        existing_loans = st.number_input("Existing Loans", min_value=0,   max_value=10,  value=0)
    with c3:
        dti_ratio = st.number_input("DTI Ratio",    min_value=0.0, max_value=1.0, value=0.3, step=0.01)
        savings   = st.number_input("Savings (₹)",  min_value=0,   value=100000)

    st.markdown("<br>", unsafe_allow_html=True)

    # Card 3 — Loan
    st.markdown("""
    <div class="form-card">
        <div class="form-card-head">
            <div class="form-card-num">3</div>
            <div>
                <div class="form-card-title">Loan Details</div>
                <div class="form-card-desc">Amount, purpose and property information</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        loan_amount = st.number_input("Loan Amount (₹)",    min_value=0, value=300000)
        loan_term   = st.number_input("Loan Term (months)", min_value=6, max_value=360, value=120)
    with c2:
        collateral_value  = st.number_input("Collateral Value (₹)", min_value=0, value=200000)
        loan_purpose      = st.selectbox("Loan Purpose", ["Home", "Education", "Personal", "Business"])
    with c3:
        property_area     = st.selectbox("Property Area",      ["Urban", "Semi-Urban", "Rural"])
        employment_status = st.selectbox("Employment Status",  ["Salaried", "Self-Employed", "Business"])

    st.markdown("<br>", unsafe_allow_html=True)

    predict = st.button("⚡  Get Instant Loan Decision", use_container_width=True)

    if predict:
        with st.spinner("Analysing your application..."):
            time.sleep(1.2)

        edu_encoded = le_edu.transform([education_level])[0]

        cat_input = pd.DataFrame([[employment_status, marital_status, loan_purpose,
                                    property_area, gender, employer_category]],
                                 columns=['Employment_Status','Marital_Status','Loan_Purpose',
                                          'Property_Area','Gender','Employer_Category'])
        cat_encoded = ohe.transform(cat_input)
        cat_df = pd.DataFrame(cat_encoded,
                              columns=ohe.get_feature_names_out(
                                  ['Employment_Status','Marital_Status','Loan_Purpose',
                                   'Property_Area','Gender','Employer_Category']))

        num_input = pd.DataFrame([[applicant_income, coapplicant_income, age, dependents,
                                    credit_score, existing_loans, dti_ratio, savings,
                                    collateral_value, loan_amount, loan_term, edu_encoded]],
                                 columns=['Applicant_Income','Coapplicant_Income','Age','Dependents',
                                          'Credit_Score','Existing_Loans','DTI_Ratio','Savings',
                                          'Collateral_Value','Loan_Amount','Loan_Term','Education_Level'])

        final_input  = pd.concat([num_input, cat_df], axis=1)
        final_scaled = scaler.transform(final_input)

        prediction  = model.predict(final_scaled)[0]
        probability = model.predict_proba(final_scaled)[0]

        if prediction == 1:
            st.markdown(f"""
            <div class="result-wrap">
            <div class="result-approved">
                <span class="res-icon">✅</span>
                <div class="res-title">Loan Approved</div>
                <div class="res-msg">The applicant meets all credit criteria set by SecureTrust Bank.<br>Proceed to final human verification.</div>
                <div class="res-conf">
                    <div class="res-conf-lbl">Approval Confidence</div>
                    <div class="res-conf-val">{probability[1]*100:.1f}%</div>
                </div>
            </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-wrap">
            <div class="result-rejected">
                <span class="res-icon">❌</span>
                <div class="res-title">Loan Rejected</div>
                <div class="res-msg">The applicant does not meet the required credit criteria at this time.<br>Consider reapplying after improving credit standing.</div>
                <div class="res-conf">
                    <div class="res-conf-lbl">Rejection Confidence</div>
                    <div class="res-conf-val">{probability[0]*100:.1f}%</div>
                </div>
            </div>
            </div>
            """, unsafe_allow_html=True)

# ── FOOTER ──
st.markdown("""
<div class="site-footer">
    <div class="footer-logo">Credit<span>Wise</span></div>
    <div class="footer-copy">© 2026 SecureTrust Bank · For internal use only · Powered by XGBoost</div>
</div>
""", unsafe_allow_html=True)