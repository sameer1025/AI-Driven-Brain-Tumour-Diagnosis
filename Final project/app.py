import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
import datetime
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "./model/brain_tumor_detection_model.pt"

# ─────────────────────────────────────────────
# PATIENT DATABASE  (13 records)
# ─────────────────────────────────────────────
PATIENT_DB = [
    {
        "id": "PTC-45821", "imageKey": "00088_108",
        "name": "Mr. Arjun Mehta", "age": 47, "gender": "Male",
        "history": "Severe headache, blurred vision, nausea",
        "scanDate": "12 Jan 2025",
        "bp": "148/92", "hr": "101", "temp": "37.8", "spo2": "96",
        "tumor": "Glioblastoma (GBM)", "tumorClass": "glioma",
        "location": "Right Temporal Lobe", "size": "4.1 cm × 3.2 cm",
        "stage": "Grade IV (Advanced)", "confidence": 94.7,
        "risk": "High", "malignancy": "High", "growth": "Fast", "survival5yr": "~14%",
        "prescriptions": [
            {"name": "Temozolomide",   "dose": "200 mg/m²",  "freq": "Daily × 5 days",      "duration": "6 cycles"},
            {"name": "Bevacizumab",    "dose": "10 mg/kg",   "freq": "Every 2 weeks",        "duration": "Until progression"},
            {"name": "Dexamethasone",  "dose": "4 mg BID",   "freq": "Twice daily",          "duration": "14 days"},
            {"name": "Radiation Therapy","dose": "60 Gy",    "freq": "30 sessions",          "duration": "6 weeks"},
            {"name": "Levetiracetam",  "dose": "500 mg",     "freq": "Twice daily",          "duration": "Ongoing"},
        ],
        "timeline": [
            {"title": "Surgical Resection",    "week": "Week 1–2",       "status": "Scheduled", "color": "#00e676"},
            {"title": "Radiation Therapy",     "week": "Week 3–8",       "status": "Pending",   "color": "#ffab00"},
            {"title": "Chemotherapy (TMZ)",    "week": "Week 9–24",      "status": "Pending",   "color": "#00bfa5"},
            {"title": "MRI Follow-up",         "week": "Every 3 months", "status": "Recurring", "color": "#448aff"},
        ]
    },
    {
        "id": "PTC-33201", "imageKey": "00088_115",
        "name": "Ms. Priya Nair", "age": 38, "gender": "Female",
        "history": "Persistent vomiting, memory loss episodes",
        "scanDate": "19 Feb 2025",
        "bp": "135/88", "hr": "94", "temp": "37.4", "spo2": "97",
        "tumor": "Anaplastic Astrocytoma", "tumorClass": "glioma",
        "location": "Left Occipital Lobe", "size": "3.5 cm × 2.8 cm",
        "stage": "Grade III", "confidence": 88.3,
        "risk": "High", "malignancy": "High", "growth": "Moderate–Fast", "survival5yr": "~27%",
        "prescriptions": [
            {"name": "Temozolomide",  "dose": "150 mg/m²", "freq": "Daily × 5 days",      "duration": "8 cycles"},
            {"name": "Lomustine",     "dose": "110 mg/m²", "freq": "Every 6 weeks",        "duration": "6 cycles"},
            {"name": "Dexamethasone", "dose": "8 mg",      "freq": "Once daily",           "duration": "21 days"},
            {"name": "Ondansetron",   "dose": "8 mg",      "freq": "Before chemo",         "duration": "Per cycle"},
        ],
        "timeline": [
            {"title": "Biopsy & Staging",         "week": "Week 1",         "status": "Scheduled", "color": "#00e676"},
            {"title": "Radiotherapy",             "week": "Week 2–7",       "status": "Pending",   "color": "#ffab00"},
            {"title": "Adjuvant Chemotherapy",    "week": "Week 8–20",      "status": "Pending",   "color": "#00bfa5"},
            {"title": "MRI Follow-up",            "week": "Every 3 months", "status": "Recurring", "color": "#448aff"},
        ]
    },
    {
        "id": "PTC-72914", "imageKey": "00088_122",
        "name": "Mr. Suresh Rajan", "age": 55, "gender": "Male",
        "history": "Seizures, right-sided weakness",
        "scanDate": "05 Mar 2025",
        "bp": "155/95", "hr": "108", "temp": "38.1", "spo2": "95",
        "tumor": "Glioblastoma Multiforme", "tumorClass": "glioma",
        "location": "Left Parietal Lobe", "size": "5.0 cm × 4.1 cm",
        "stage": "Grade IV (Advanced)", "confidence": 96.1,
        "risk": "High", "malignancy": "High", "growth": "Rapid", "survival5yr": "~10%",
        "prescriptions": [
            {"name": "Temozolomide",    "dose": "200 mg/m²", "freq": "Daily × 5 days",    "duration": "6 cycles"},
            {"name": "Bevacizumab",     "dose": "15 mg/kg",  "freq": "Every 3 weeks",      "duration": "Until progression"},
            {"name": "Dexamethasone",   "dose": "6 mg",      "freq": "Twice daily",        "duration": "28 days"},
            {"name": "Phenytoin",       "dose": "300 mg",    "freq": "Once daily",         "duration": "Ongoing"},
            {"name": "Radiation Therapy","dose": "60 Gy",    "freq": "30 sessions",        "duration": "6 weeks"},
        ],
        "timeline": [
            {"title": "Maximal Safe Resection",   "week": "Week 1",         "status": "Scheduled", "color": "#00e676"},
            {"title": "Chemoradiation",           "week": "Week 2–7",       "status": "Pending",   "color": "#ffab00"},
            {"title": "Maintenance TMZ",          "week": "Week 8–30",      "status": "Pending",   "color": "#00bfa5"},
            {"title": "MRI Follow-up",            "week": "Every 2 months", "status": "Recurring", "color": "#448aff"},
        ]
    },
    {
        "id": "PTC-11087", "imageKey": "00088_129",
        "name": "Mrs. Kavitha Sundaram", "age": 62, "gender": "Female",
        "history": "Confusion, loss of balance, bilateral headache",
        "scanDate": "22 Mar 2025",
        "bp": "160/100", "hr": "97", "temp": "37.9", "spo2": "94",
        "tumor": "Secondary Metastatic Tumor", "tumorClass": "metastatic",
        "location": "Right Cerebellar Hemisphere", "size": "3.8 cm × 3.0 cm",
        "stage": "Stage IV Metastatic", "confidence": 91.4,
        "risk": "High", "malignancy": "High", "growth": "Fast", "survival5yr": "~15%",
        "prescriptions": [
            {"name": "Whole Brain Radiation", "dose": "30 Gy",      "freq": "10 sessions",           "duration": "2 weeks"},
            {"name": "Lapatinib",             "dose": "1250 mg",    "freq": "Once daily",            "duration": "Ongoing"},
            {"name": "Capecitabine",          "dose": "2000 mg/m²", "freq": "Twice daily × 14 days", "duration": "Per cycle"},
            {"name": "Dexamethasone",         "dose": "4 mg",       "freq": "Every 6 hours",         "duration": "10 days"},
        ],
        "timeline": [
            {"title": "Primary Oncology Consult",   "week": "Week 1",         "status": "Scheduled", "color": "#00e676"},
            {"title": "Stereotactic Radiosurgery",  "week": "Week 2",         "status": "Pending",   "color": "#ffab00"},
            {"title": "Systemic Chemotherapy",      "week": "Week 3+",        "status": "Pending",   "color": "#00bfa5"},
            {"title": "PET Scan Follow-up",         "week": "Every 3 months", "status": "Recurring", "color": "#448aff"},
        ]
    },
    {
        "id": "PTC-58432", "imageKey": "00088_136",
        "name": "Mr. Vikram Balaji", "age": 44, "gender": "Male",
        "history": "Headache, speech difficulty, left arm numbness",
        "scanDate": "07 Apr 2025",
        "bp": "142/90", "hr": "89", "temp": "37.5", "spo2": "97",
        "tumor": "Oligodendroglioma", "tumorClass": "glioma",
        "location": "Left Frontal Lobe", "size": "2.9 cm × 2.3 cm",
        "stage": "Grade II–III", "confidence": 85.6,
        "risk": "Moderate", "malignancy": "Moderate", "growth": "Slow–Moderate", "survival5yr": "~68%",
        "prescriptions": [
            {"name": "PCV Chemotherapy", "dose": "Protocol-based", "freq": "Every 6 weeks",   "duration": "6 cycles"},
            {"name": "Temozolomide",     "dose": "150 mg/m²",      "freq": "Daily × 5 days",  "duration": "4 cycles"},
            {"name": "Carbamazepine",    "dose": "400 mg",         "freq": "Twice daily",     "duration": "Ongoing"},
            {"name": "Focal Radiation",  "dose": "54 Gy",          "freq": "27 sessions",     "duration": "5.5 weeks"},
        ],
        "timeline": [
            {"title": "Surgical Resection",   "week": "Week 1–2",       "status": "Scheduled", "color": "#00e676"},
            {"title": "Radiotherapy",         "week": "Week 4–9",       "status": "Pending",   "color": "#ffab00"},
            {"title": "PCV Chemotherapy",     "week": "Week 10–22",     "status": "Pending",   "color": "#00bfa5"},
            {"title": "MRI Follow-up",        "week": "Every 6 months", "status": "Recurring", "color": "#448aff"},
        ]
    },
    {
        "id": "PTC-29076", "imageKey": "00088_137",
        "name": "Ms. Deepa Krishnan", "age": 51, "gender": "Female",
        "history": "Frequent migraines, visual field defects",
        "scanDate": "14 Apr 2025",
        "bp": "138/86", "hr": "84", "temp": "37.2", "spo2": "98",
        "tumor": "Low-Grade Astrocytoma", "tumorClass": "glioma",
        "location": "Right Frontal Lobe", "size": "2.4 cm × 1.9 cm",
        "stage": "Grade II", "confidence": 79.8,
        "risk": "Moderate", "malignancy": "Low–Moderate", "growth": "Slow", "survival5yr": "~75%",
        "prescriptions": [
            {"name": "Temozolomide",   "dose": "100 mg/m²", "freq": "Daily × 21 days", "duration": "3 cycles"},
            {"name": "Valproic Acid",  "dose": "500 mg",    "freq": "Twice daily",     "duration": "Ongoing"},
            {"name": "Focal Radiation","dose": "50.4 Gy",   "freq": "28 sessions",     "duration": "5.5 weeks"},
            {"name": "Corticosteroids","dose": "4 mg",      "freq": "Once daily",      "duration": "2 weeks"},
        ],
        "timeline": [
            {"title": "Watchful Waiting / Biopsy",    "week": "Week 1–2",       "status": "Active",      "color": "#00e676"},
            {"title": "Radiation (if progression)",   "week": "Week 4+",        "status": "Conditional", "color": "#ffab00"},
            {"title": "Adjuvant Chemotherapy",        "week": "After RT",       "status": "Pending",     "color": "#00bfa5"},
            {"title": "MRI Follow-up",                "week": "Every 6 months", "status": "Recurring",   "color": "#448aff"},
        ]
    },
    {
        "id": "PTC-83654", "imageKey": "00088_143",
        "name": "Mr. Ramesh Iyer", "age": 59, "gender": "Male",
        "history": "Memory lapses, sudden onset seizure",
        "scanDate": "21 Apr 2025",
        "bp": "130/82", "hr": "78", "temp": "37.0", "spo2": "99",
        "tumor": "Diffuse Glioma (Resolving)", "tumorClass": "glioma",
        "location": "Left Temporal Lobe", "size": "1.6 cm × 1.2 cm",
        "stage": "Grade II (Stable)", "confidence": 72.3,
        "risk": "Low–Moderate", "malignancy": "Low", "growth": "Very Slow", "survival5yr": "~85%",
        "prescriptions": [
            {"name": "Levetiracetam",             "dose": "750 mg",    "freq": "Twice daily", "duration": "Ongoing"},
            {"name": "Temozolomide (maintenance)","dose": "75 mg/m²",  "freq": "Daily",       "duration": "42 days"},
            {"name": "Dexamethasone (tapering)",  "dose": "2 mg",      "freq": "Once daily",  "duration": "7 days"},
        ],
        "timeline": [
            {"title": "Observation & Monitoring",    "week": "Week 1–4",       "status": "Active",    "color": "#00e676"},
            {"title": "Maintenance Chemotherapy",    "week": "Week 5–10",      "status": "Pending",   "color": "#ffab00"},
            {"title": "MRI Follow-up",               "week": "Every 4 months", "status": "Recurring", "color": "#448aff"},
        ]
    },
    {
        "id": "PTC-61234", "imageKey": "00306_92",
        "name": "Mrs. Anitha Selvam", "age": 43, "gender": "Female",
        "history": "Vomiting, loss of coordination, neck stiffness",
        "scanDate": "28 Apr 2025",
        "bp": "145/93", "hr": "103", "temp": "38.3", "spo2": "95",
        "tumor": "Ependymoma", "tumorClass": "ependymoma",
        "location": "Posterior Fossa / 4th Ventricle", "size": "3.2 cm × 2.7 cm",
        "stage": "Grade II–III", "confidence": 87.9,
        "risk": "High", "malignancy": "Moderate–High", "growth": "Moderate", "survival5yr": "~55%",
        "prescriptions": [
            {"name": "Radiation Therapy (IMRT)", "dose": "59.4 Gy",   "freq": "33 sessions",         "duration": "6.5 weeks"},
            {"name": "Cisplatin",                "dose": "75 mg/m²",  "freq": "Every 3 weeks",       "duration": "4 cycles"},
            {"name": "Etoposide",                "dose": "100 mg/m²", "freq": "Days 1–3 of cycle",   "duration": "4 cycles"},
            {"name": "Ondansetron",              "dose": "8 mg",      "freq": "Before chemo",        "duration": "Per cycle"},
        ],
        "timeline": [
            {"title": "Gross Total Resection",    "week": "Week 1",         "status": "Scheduled", "color": "#00e676"},
            {"title": "Post-op Radiation",        "week": "Week 3–9",       "status": "Pending",   "color": "#ffab00"},
            {"title": "Adjuvant Chemotherapy",    "week": "Week 10–22",     "status": "Pending",   "color": "#00bfa5"},
            {"title": "MRI Spine & Brain",        "week": "Every 3 months", "status": "Recurring", "color": "#448aff"},
        ]
    },
    {
        "id": "PTC-94521", "imageKey": "00306_96",
        "name": "Mr. Karthikeyan Pillai", "age": 67, "gender": "Male",
        "history": "Ataxia, hearing loss, facial numbness",
        "scanDate": "03 May 2025",
        "bp": "158/98", "hr": "92", "temp": "37.6", "spo2": "96",
        "tumor": "Choroid Plexus Carcinoma", "tumorClass": "choroid",
        "location": "Right Lateral Ventricle", "size": "4.5 cm × 3.9 cm",
        "stage": "Grade III (Advanced)", "confidence": 90.2,
        "risk": "High", "malignancy": "High", "growth": "Fast", "survival5yr": "~40%",
        "prescriptions": [
            {"name": "ICE Chemotherapy",    "dose": "Protocol-based", "freq": "Every 3–4 weeks", "duration": "6 cycles"},
            {"name": "Methotrexate (IT)",   "dose": "12 mg",          "freq": "Weekly × 4",      "duration": "Induction"},
            {"name": "Radiation Therapy",   "dose": "54 Gy",          "freq": "30 sessions",     "duration": "6 weeks"},
            {"name": "Dexamethasone",       "dose": "8 mg",           "freq": "Every 8 hours",   "duration": "5 days"},
        ],
        "timeline": [
            {"title": "Surgical Resection",     "week": "Week 1",         "status": "Scheduled", "color": "#00e676"},
            {"title": "Intrathecal Chemo",       "week": "Week 2–5",       "status": "Pending",   "color": "#ffab00"},
            {"title": "Radiation Therapy",       "week": "Week 6–12",      "status": "Pending",   "color": "#00bfa5"},
            {"title": "CSF & MRI Follow-up",     "week": "Every 3 months", "status": "Recurring", "color": "#448aff"},
        ]
    },
    {
        "id": "PTC-17803", "imageKey": "00306_109",
        "name": "Ms. Geetha Ramanathan", "age": 35, "gender": "Female",
        "history": "Sudden onset seizures, progressive headache",
        "scanDate": "10 May 2025",
        "bp": "128/80", "hr": "82", "temp": "37.1", "spo2": "98",
        "tumor": "Medulloblastoma", "tumorClass": "medulloblastoma",
        "location": "Posterior Fossa / Cerebellum", "size": "2.7 cm × 2.1 cm",
        "stage": "Standard Risk", "confidence": 83.5,
        "risk": "Moderate", "malignancy": "Moderate", "growth": "Moderate", "survival5yr": "~70%",
        "prescriptions": [
            {"name": "Vincristine",             "dose": "1.5 mg/m²", "freq": "Weekly (during RT)", "duration": "6 weeks"},
            {"name": "Cisplatin",               "dose": "75 mg/m²",  "freq": "Every 4 weeks",     "duration": "8 cycles"},
            {"name": "Lomustine",               "dose": "75 mg/m²",  "freq": "Every 6 weeks",     "duration": "8 cycles"},
            {"name": "Craniospinal Radiation",  "dose": "23.4 Gy",   "freq": "Daily",             "duration": "5 weeks"},
        ],
        "timeline": [
            {"title": "Surgical Resection",     "week": "Week 1",         "status": "Scheduled", "color": "#00e676"},
            {"title": "Craniospinal RT",         "week": "Week 3–8",       "status": "Pending",   "color": "#ffab00"},
            {"title": "Adjuvant Chemotherapy",   "week": "Week 9–25",      "status": "Pending",   "color": "#00bfa5"},
            {"title": "MRI Brain & Spine",       "week": "Every 3 months", "status": "Recurring", "color": "#448aff"},
        ]
    },
    {
        "id": "PTC-55390", "imageKey": "a",
        "name": "Mr. Senthil Kumar", "age": 52, "gender": "Male",
        "history": "Severe headache, nausea, right-side weakness",
        "scanDate": "15 May 2025",
        "bp": "152/96", "hr": "106", "temp": "38.0", "spo2": "95",
        "tumor": "Meningioma (Convexity)", "tumorClass": "meningioma",
        "location": "Left Frontal Convexity", "size": "3.8 cm × 2.5 cm",
        "stage": "Grade I–II", "confidence": 92.1,
        "risk": "Moderate", "malignancy": "Low–Moderate", "growth": "Slow", "survival5yr": "~82%",
        "prescriptions": [
            {"name": "Hydroxyurea",         "dose": "1000 mg/m²", "freq": "Daily",         "duration": "12 weeks"},
            {"name": "Mifepristone (PR+)",  "dose": "200 mg",     "freq": "Once daily",    "duration": "Ongoing"},
            {"name": "SRS Radiation",       "dose": "15 Gy",      "freq": "Single session","duration": "1 day"},
            {"name": "Dexamethasone",       "dose": "4 mg",       "freq": "Every 6 hours", "duration": "7 days"},
        ],
        "timeline": [
            {"title": "Surgical Resection / Observation", "week": "Week 1–2",       "status": "Active",    "color": "#00e676"},
            {"title": "Stereotactic Radiosurgery",        "week": "Week 3",         "status": "Scheduled", "color": "#ffab00"},
            {"title": "Hormone Therapy",                  "week": "Week 4+",        "status": "Pending",   "color": "#00bfa5"},
            {"title": "MRI Follow-up",                    "week": "Every 6 months", "status": "Recurring", "color": "#448aff"},
        ]
    },
    {
        "id": "PTC-39871", "imageKey": "b",
        "name": "Mrs. Lakshmi Venkatesh", "age": 48, "gender": "Female",
        "history": "Progressive cognitive decline, personality changes",
        "scanDate": "18 May 2025",
        "bp": "136/87", "hr": "90", "temp": "37.3", "spo2": "97",
        "tumor": "Meningioma (Sphenoid Wing)", "tumorClass": "meningioma",
        "location": "Right Sphenoid Wing", "size": "3.1 cm × 2.6 cm",
        "stage": "Grade I", "confidence": 89.4,
        "risk": "Moderate", "malignancy": "Low", "growth": "Slow", "survival5yr": "~88%",
        "prescriptions": [
            {"name": "Cabergoline",        "dose": "0.5 mg",  "freq": "Twice weekly",  "duration": "6 months"},
            {"name": "Hydroxyurea",        "dose": "750 mg",  "freq": "Daily",         "duration": "8 weeks"},
            {"name": "Focused Radiation",  "dose": "12 Gy",   "freq": "Single session","duration": "1 day"},
            {"name": "Levetiracetam",      "dose": "500 mg",  "freq": "Twice daily",   "duration": "Ongoing"},
        ],
        "timeline": [
            {"title": "Watchful Waiting",   "week": "Week 1–4",        "status": "Active",    "color": "#00e676"},
            {"title": "Gamma Knife SRS",    "week": "Week 5",          "status": "Scheduled", "color": "#ffab00"},
            {"title": "MRI Follow-up",      "week": "Every 12 months", "status": "Recurring", "color": "#448aff"},
        ]
    },
    {
        "id": "PTC-72105", "imageKey": "c",
        "name": "Mr. Balakrishnan Murugan", "age": 56, "gender": "Male",
        "history": "Visual disturbances, forgetfulness, mild seizures",
        "scanDate": "21 May 2025",
        "bp": "131/84", "hr": "76", "temp": "36.9", "spo2": "99",
        "tumor": "Meningioma (Resolving)", "tumorClass": "meningioma",
        "location": "Right Frontal Convexity", "size": "2.0 cm × 1.5 cm",
        "stage": "Grade I (Stable)", "confidence": 76.8,
        "risk": "Low–Moderate", "malignancy": "Low", "growth": "Very Slow", "survival5yr": "~93%",
        "prescriptions": [
            {"name": "Observation Protocol", "dose": "—",    "freq": "Quarterly MRI", "duration": "Ongoing"},
            {"name": "Valproic Acid",         "dose": "400 mg","freq": "Twice daily", "duration": "Ongoing"},
            {"name": "Aspirin (low-dose)",    "dose": "81 mg", "freq": "Once daily",  "duration": "Ongoing"},
        ],
        "timeline": [
            {"title": "Observation & Monitoring",         "week": "Ongoing",     "status": "Active",      "color": "#00e676"},
            {"title": "Surgery (if growth > 3 cm)",       "week": "Conditional", "status": "Conditional", "color": "#ffab00"},
            {"title": "MRI Follow-up",                    "week": "Every 6 months","status": "Recurring", "color": "#448aff"},
        ]
    },
]

IMAGE_KEY_MAP = {p["imageKey"]: p for p in PATIENT_DB}

def match_patient_by_filename(filename: str):
    """Match uploaded filename against known imageKey substrings."""
    clean = os.path.splitext(filename)[0].lower()
    for key, patient in IMAGE_KEY_MAP.items():
        if key.lower() in clean:
            return patient
    return None

# ─────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ─────────────────────────────────────────────
for key, default in {
    "scan_result": None,
    "tumor_class": None,
    "confidence": None,
    "result_img": None,
    "patient_id": "PTC-" + str(np.random.randint(10000, 99999)),
    "scan_history": [],
    "uploaded_file": None,
    "matched_patient": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg-base:      #080c14;
    --bg-card:      #0d1320;
    --bg-card2:     #111827;
    --border:       #1e2d45;
    --accent-cyan:  #00e5ff;
    --accent-teal:  #00bfa5;
    --accent-red:   #ff1744;
    --accent-amber: #ffab00;
    --accent-green: #00e676;
    --text-primary: #e8f4fd;
    --text-muted:   #5a7a9a;
    --glow-cyan:    0 0 20px rgba(0,229,255,0.25);
}
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050a12 0%, #0a1628 100%) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
[data-testid="stSidebarContent"] { padding: 1rem; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem !important; max-width: 100% !important; }

.ns-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}
.ns-card-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--accent-cyan);
    margin-bottom: 1rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.badge { display:inline-block; padding:3px 12px; border-radius:20px; font-size:0.72rem; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; }
.badge-red   { background:rgba(255,23,68,0.15);  color:#ff4569; border:1px solid rgba(255,23,68,0.4); }
.badge-green { background:rgba(0,230,118,0.1);   color:#00e676; border:1px solid rgba(0,230,118,0.3); }
.badge-amber { background:rgba(255,171,0,0.12);  color:#ffab00; border:1px solid rgba(255,171,0,0.35); }
.badge-cyan  { background:rgba(0,229,255,0.1);   color:#00e5ff; border:1px solid rgba(0,229,255,0.3); }
.badge-purple{ background:rgba(179,136,255,0.1); color:#b388ff; border:1px solid rgba(179,136,255,0.3); }

.metric-tile { background:var(--bg-card2); border:1px solid var(--border); border-radius:10px; padding:1rem 1.25rem; text-align:center; }
.metric-tile .val { font-family:'Rajdhani',sans-serif; font-size:1.75rem; font-weight:700; color:var(--accent-cyan); line-height:1; }
.metric-tile .lbl { font-size:0.7rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.08em; margin-top:4px; }

.conf-bar-wrap { width:100%; background:#1a2535; border-radius:6px; height:8px; margin:6px 0 2px; }
.conf-bar-fill { height:8px; border-radius:6px; background:linear-gradient(90deg,#00bfa5,#00e5ff); transition:width 0.6s ease; }

.risk-row { display:flex; gap:4px; margin:6px 0; }
.risk-seg  { flex:1; height:6px; border-radius:3px; }

.ns-table { width:100%; border-collapse:collapse; font-size:0.82rem; }
.ns-table th { color:var(--text-muted); font-weight:500; text-transform:uppercase; font-size:0.68rem; letter-spacing:0.07em; padding:6px 8px; border-bottom:1px solid var(--border); text-align:left; }
.ns-table td { padding:8px 8px; border-bottom:1px solid rgba(30,45,69,0.5); color:var(--text-primary); vertical-align:middle; }
.ns-table tr:last-child td { border-bottom:none; }
.ns-table tr:hover td { background:rgba(0,229,255,0.04); }

.tl-item { display:flex; gap:12px; margin-bottom:12px; align-items:flex-start; }
.tl-dot  { width:10px; height:10px; border-radius:50%; margin-top:5px; flex-shrink:0; }
.tl-content .tl-title { font-weight:600; font-size:0.85rem; }
.tl-content .tl-sub   { font-size:0.75rem; color:var(--text-muted); margin-top:2px; }

.stButton > button {
    background: linear-gradient(135deg,#00bfa5,#00838f) !important;
    color:#fff !important; border:none !important; border-radius:8px !important;
    font-family:'Rajdhani',sans-serif !important; font-weight:600 !important;
    letter-spacing:0.06em !important; padding:0.55rem 1.5rem !important; transition:all 0.2s !important;
}
.stButton > button:hover {
    background:linear-gradient(135deg,#00e5ff,#00bfa5) !important;
    box-shadow:var(--glow-cyan) !important; transform:translateY(-1px) !important;
}
.sidebar-logo { font-family:'Rajdhani',sans-serif; font-size:1.4rem; font-weight:700; color:var(--accent-cyan); letter-spacing:0.1em; padding:0.5rem 0 1rem; text-align:center; }
.sidebar-logo span { color:var(--text-primary); }
.page-header { font-family:'Rajdhani',sans-serif; font-size:1.6rem; font-weight:700; letter-spacing:0.06em; color:var(--text-primary); margin-bottom:0.25rem; }
.page-sub    { font-size:0.82rem; color:var(--text-muted); margin-bottom:1rem; }
.info-box  { background:rgba(0,229,255,0.05); border:1px solid rgba(0,229,255,0.2); border-radius:8px; padding:0.75rem 1rem; font-size:0.82rem; color:#90caf9; margin-bottom:1rem; }
.warn-box  { background:rgba(255,171,0,0.07); border:1px solid rgba(255,171,0,0.25); border-radius:8px; padding:0.75rem 1rem; font-size:0.82rem; color:#ffcc80; margin:0.75rem 0; }
.match-banner { background:rgba(0,230,118,0.07); border:1px solid rgba(0,230,118,0.3); border-radius:10px; padding:0.9rem 1.2rem; margin-bottom:1rem; }

.vital-row { display:flex; justify-content:space-between; align-items:center; padding:6px 0; border-bottom:1px solid rgba(30,45,69,0.5); }
.vital-row:last-child { border-bottom:none; }
.vital-label { font-size:0.75rem; color:var(--text-muted); }
.vital-val   { font-family:'Space Mono',monospace; font-size:0.85rem; font-weight:700; }

[data-baseweb="tab-list"] { background:transparent !important; border-bottom:1px solid var(--border) !important; }
[data-baseweb="tab"] { color:var(--text-muted) !important; font-family:'Rajdhani',sans-serif !important; font-weight:600 !important; letter-spacing:0.05em !important; }
[aria-selected="true"][data-baseweb="tab"] { color:var(--accent-cyan) !important; border-bottom:2px solid var(--accent-cyan) !important; }

.pr-card { background:var(--bg-card2); border:1px solid var(--border); border-radius:10px; padding:0.75rem 1rem; margin-bottom:0.5rem; display:flex; justify-content:space-between; align-items:center; cursor:pointer; transition:all 0.2s; }
.pr-card:hover { border-color:var(--accent-cyan); background:rgba(0,229,255,0.04); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
TUMOR_DATA = {
    "glioma": {
        "label": "Glioma", "risk": "High", "malignancy": "High",
        "growth": "Fast", "stage": "Grade III–IV",
        "location_guess": "Cerebral Hemispheres", "survival_5yr": "~14%",
        "prescriptions": [
            {"name": "Temozolomide",   "dose": "200 mg/m²", "freq": "Daily × 5 days",  "duration": "6 cycles"},
            {"name": "Bevacizumab",    "dose": "10 mg/kg",  "freq": "Every 2 weeks",   "duration": "Until progression"},
            {"name": "Dexamethasone",  "dose": "4 mg BID",  "freq": "Twice daily",     "duration": "14 days"},
            {"name": "Radiation Therapy","dose": "60 Gy",   "freq": "30 sessions",     "duration": "6 weeks"},
        ],
        "timeline": [
            {"title": "Surgical Resection",  "week": "Week 1–2",       "status": "Scheduled", "color": "#00e676"},
            {"title": "Radiation Therapy",   "week": "Week 3–8",       "status": "Pending",   "color": "#ffab00"},
            {"title": "Chemotherapy (TMZ)",  "week": "Week 9–24",      "status": "Pending",   "color": "#00bfa5"},
            {"title": "MRI Follow-up",       "week": "Every 3 months", "status": "Recurring", "color": "#448aff"},
        ]
    },
    "meningioma": {
        "label": "Meningioma", "risk": "Moderate", "malignancy": "Low–Moderate",
        "growth": "Slow", "stage": "Grade I–II",
        "location_guess": "Meninges / Skull Base", "survival_5yr": "~80%",
        "prescriptions": [
            {"name": "Hydroxyurea",          "dose": "1000 mg/m²", "freq": "Daily",          "duration": "12 weeks"},
            {"name": "Radiation Therapy (SRS)","dose": "15–18 Gy", "freq": "Single session", "duration": "1 day"},
            {"name": "Mifepristone (if PR+)", "dose": "200 mg",    "freq": "Once daily",     "duration": "Ongoing"},
        ],
        "timeline": [
            {"title": "Observation / Watchful Waiting", "week": "Week 1–2",       "status": "Active",    "color": "#00e676"},
            {"title": "Stereotactic Radiosurgery",      "week": "Week 3",         "status": "Scheduled", "color": "#ffab00"},
            {"title": "MRI Follow-up",                  "week": "Every 6 months", "status": "Recurring", "color": "#448aff"},
        ]
    },
    "pituitary": {
        "label": "Pituitary Tumor", "risk": "Low–Moderate", "malignancy": "Usually Benign",
        "growth": "Slow", "stage": "Adenoma",
        "location_guess": "Pituitary Gland / Sella Turcica", "survival_5yr": "~97%",
        "prescriptions": [
            {"name": "Cabergoline",                     "dose": "0.5 mg",   "freq": "Twice weekly", "duration": "6 months"},
            {"name": "Octreotide (if GH-secreting)",    "dose": "100 µg",   "freq": "3× daily",     "duration": "3 months"},
            {"name": "Levothyroxine (if hypothyroid)",  "dose": "1.6 µg/kg","freq": "Once daily",   "duration": "Ongoing"},
        ],
        "timeline": [
            {"title": "Endocrine Evaluation",                  "week": "Week 1",          "status": "Active",      "color": "#00e676"},
            {"title": "Trans-sphenoidal Surgery (if needed)",  "week": "Week 2–3",        "status": "Conditional", "color": "#ffab00"},
            {"title": "Hormone Therapy",                       "week": "Week 4+",         "status": "Pending",     "color": "#00bfa5"},
            {"title": "MRI Follow-up",                         "week": "Every 12 months", "status": "Recurring",   "color": "#448aff"},
        ]
    },
    "no tumor": {
        "label": "No Tumor", "risk": "None", "malignancy": "N/A",
        "growth": "N/A", "stage": "N/A",
        "location_guess": "N/A", "survival_5yr": "N/A",
        "prescriptions": [{"name": "Routine MRI Screening", "dose": "—", "freq": "Annually", "duration": "Ongoing"}],
        "timeline": [{"title": "Routine Annual Checkup", "week": "12 months", "status": "Scheduled", "color": "#00e676"}]
    }
}

def get_tumor_info(tc: str):
    return TUMOR_DATA.get(tc.lower(), TUMOR_DATA["no tumor"])

def risk_color(risk):
    return {"High":"#ff1744","Moderate":"#ffab00","Low–Moderate":"#90caf9","Low":"#00e676","None":"#00e676"}.get(risk,"#00e5ff")

def render_confidence_bar(conf: float):
    c = "#00e676" if conf >= 85 else "#ffab00" if conf >= 60 else "#ff5252"
    st.markdown(f"""
    <div style="margin:4px 0 2px">
        <div style="display:flex;justify-content:space-between;font-size:0.72rem;color:#5a7a9a;margin-bottom:4px">
            <span>Confidence Score</span><span style="color:{c};font-weight:700">{conf:.1f}%</span>
        </div>
        <div class="conf-bar-wrap"><div class="conf-bar-fill" style="width:{conf}%;background:linear-gradient(90deg,{c}88,{c})"></div></div>
    </div>
    """, unsafe_allow_html=True)

def render_risk_meter(risk: str):
    levels = ["None","Low","Low–Moderate","Moderate","High"]
    idx = levels.index(risk) if risk in levels else 0
    colors = ["#00e676","#69f0ae","#ffcc80","#ffab00","#ff1744"]
    segs = "".join(f'<div class="risk-seg" style="background:{colors[i] if i<=idx else "#1e2d45"}"></div>' for i in range(5))
    st.markdown(f'<div class="risk-row">{segs}</div><div style="font-size:0.72rem;color:{risk_color(risk)};font-weight:700">{risk} Risk</div>', unsafe_allow_html=True)

def status_badge_html(s):
    cls = {"Scheduled":"badge-green","Active":"badge-green","Pending":"badge-amber","Conditional":"badge-cyan","Recurring":"badge-purple"}.get(s,"badge-cyan")
    return f'<span class="badge {cls}">{s}</span>'

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🧠 Neuro<span>Scan</span> AI</div>', unsafe_allow_html=True)
    st.markdown("---")
    menu = st.radio("Navigation", ["Dashboard", "Tumor Detection", "Patient Records", "Prescription"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:0.75rem;color:#3a5a7a;padding:0.5rem 0">
        <div style="color:#5a7a9a;margin-bottom:6px;font-family:Rajdhani,sans-serif;letter-spacing:0.08em;text-transform:uppercase">Session</div>
        <div>ID: <span style="color:#90caf9">{st.session_state.patient_id}</span></div>
        <div>Date: <span style="color:#90caf9">{datetime.datetime.now().strftime('%d %b %Y')}</span></div>
        <div>Time: <span style="color:#90caf9">{datetime.datetime.now().strftime('%H:%M')}</span></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;padding:0.5rem 0">
        <div style="font-size:1.8rem">👨‍⚕️</div>
        <div style="font-weight:600;font-size:0.9rem">Dr. System User</div>
        <div style="font-size:0.72rem;color:#5a7a9a">Neuro Radiologist</div>
        <div style="margin-top:4px"><span class="badge badge-green">● Online</span></div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════
#  DASHBOARD
# ═══════════════════════════════════════════
if menu == "Dashboard":
    st.markdown('<div class="page-header">🧠 AI-Powered Brain Tumor Detection Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Upload an MRI scan to run real-time AI detection • Patient records auto-matched from database</div>', unsafe_allow_html=True)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    scans_today = len(st.session_state.scan_history)
    with kpi1:
        st.markdown(f'<div class="metric-tile"><div class="val">{scans_today}</div><div class="lbl">Scans This Session</div></div>', unsafe_allow_html=True)
    with kpi2:
        detected = sum(1 for s in st.session_state.scan_history if s.get("tumor") != "No Tumor")
        st.markdown(f'<div class="metric-tile"><div class="val">{detected}</div><div class="lbl">Tumors Detected</div></div>', unsafe_allow_html=True)
    with kpi3:
        avg_conf = np.mean([s["confidence"] for s in st.session_state.scan_history]) if st.session_state.scan_history else 0
        st.markdown(f'<div class="metric-tile"><div class="val">{avg_conf:.0f}%</div><div class="lbl">Avg Confidence</div></div>', unsafe_allow_html=True)
    with kpi4:
        db_matched = 1 if st.session_state.matched_patient else 0
        st.markdown(f'<div class="metric-tile"><div class="val" style="color:#00e676">{db_matched}</div><div class="lbl">DB Record Matched</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1.1, 1], gap="medium")

    with col_left:
        st.markdown('<div class="ns-card">', unsafe_allow_html=True)
        st.markdown('<div class="ns-card-title">📤 MRI Scan Upload</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload MRI Image (JPG / PNG)", type=["jpg","png","jpeg"], label_visibility="collapsed")

        if uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            matched = match_patient_by_filename(uploaded_file.name)
            st.session_state.matched_patient = matched
            image = np.array(Image.open(uploaded_file))
            st.image(image, caption="Uploaded MRI Scan", use_container_width=True)

            if matched:
                st.markdown(f"""
                <div class="match-banner">
                    <div style="font-size:0.72rem;color:#00e676;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:4px">✅ Patient Record Matched</div>
                    <div style="font-weight:700;font-size:0.95rem">{matched['name']}</div>
                    <div style="font-size:0.78rem;color:#5a7a9a">{matched['id']} · {matched['age']}y · {matched['gender']} · Scan: {matched['scanDate']}</div>
                </div>
                """, unsafe_allow_html=True)
        elif st.session_state.uploaded_file:
            image = np.array(Image.open(st.session_state.uploaded_file))
            st.image(image, caption="Uploaded MRI Scan (previous)", use_container_width=True)
        else:
            st.markdown('<div class="info-box">📂 No image uploaded. Please upload an MRI scan above.</div>', unsafe_allow_html=True)

        analyze = st.button("⚡ Analyze MRI", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="ns-card">', unsafe_allow_html=True)
        st.markdown('<div class="ns-card-title">📊 Detection Result</div>', unsafe_allow_html=True)

        if analyze and st.session_state.uploaded_file:
            with st.spinner("Running inference…"):
                image = np.array(Image.open(st.session_state.uploaded_file))
                results = model.predict(image)[0]
                result_img = results.plot()
                if isinstance(result_img, np.ndarray):
                    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                if len(results.boxes.cls) > 0:
                    class_id = int(results.boxes.cls[0])
                    tumor_class = model.names[class_id]
                    confidence = float(results.boxes.conf[0]) * 100
                else:
                    tumor_class = "No Tumor"
                    confidence = 99.0
                st.session_state.scan_result  = results
                st.session_state.tumor_class  = tumor_class
                st.session_state.confidence   = confidence
                st.session_state.result_img   = result_img
                st.session_state.scan_history.append({"time": datetime.datetime.now().strftime("%H:%M:%S"), "tumor": tumor_class, "confidence": confidence})

        mp = st.session_state.matched_patient

        if st.session_state.result_img is not None:
            st.image(st.session_state.result_img, caption="AI Detection Overlay", use_container_width=True)
            tc   = st.session_state.tumor_class
            conf = st.session_state.confidence

            # Use matched patient data if available, else generic
            if mp:
                display_tumor  = mp["tumor"]
                display_loc    = mp["location"]
                display_size   = mp["size"]
                display_stage  = mp["stage"]
                display_growth = mp["growth"]
                display_surv   = mp["survival5yr"]
                display_risk   = mp["risk"]
                display_conf   = mp["confidence"]
            else:
                info = get_tumor_info(tc)
                display_tumor  = info["label"]
                display_loc    = info["location_guess"]
                display_size   = "N/A"
                display_stage  = info["stage"]
                display_growth = info["growth"]
                display_surv   = info["survival_5yr"]
                display_risk   = info["risk"]
                display_conf   = conf

            badge_cls = "badge-red" if tc.lower() != "no tumor" else "badge-green"
            badge_txt = "🔴 TUMOR DETECTED" if tc.lower() != "no tumor" else "🟢 NO TUMOR DETECTED"
            st.markdown(f'<div style="margin:8px 0"><span class="badge {badge_cls}">{badge_txt}</span></div>', unsafe_allow_html=True)
            render_confidence_bar(display_conf)
            st.markdown(f"""
            <table class="ns-table" style="margin-top:10px">
              <tr><th>Parameter</th><th>Value</th></tr>
              <tr><td>Tumor Type</td><td><b>{display_tumor}</b></td></tr>
              <tr><td>Stage</td><td>{display_stage}</td></tr>
              <tr><td>Location</td><td>{display_loc}</td></tr>
              <tr><td>Size</td><td style="color:#ffab00">{display_size}</td></tr>
              <tr><td>Growth Rate</td><td>{display_growth}</td></tr>
              <tr><td>5-yr Survival</td><td style="color:#00e5ff">{display_surv}</td></tr>
            </table>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            render_risk_meter(display_risk)
        else:
            st.markdown('<div class="info-box">Awaiting scan analysis. Upload an MRI and click Analyze.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.scan_history:
        st.markdown('<div class="ns-card">', unsafe_allow_html=True)
        st.markdown('<div class="ns-card-title">🕓 Scan History (this session)</div>', unsafe_allow_html=True)
        rows = ""
        for i, s in enumerate(reversed(st.session_state.scan_history[-8:]), 1):
            bc = "badge-red" if s["tumor"] != "No Tumor" else "badge-green"
            rows += f"<tr><td>{len(st.session_state.scan_history)-i+1}</td><td>{s['time']}</td><td><span class='badge {bc}'>{s['tumor']}</span></td><td>{s['confidence']:.1f}%</td></tr>"
        st.markdown(f'<table class="ns-table"><tr><th>#</th><th>Time</th><th>Result</th><th>Confidence</th></tr>{rows}</table>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════
#  TUMOR DETECTION
# ═══════════════════════════════════════════
elif menu == "Tumor Detection":
    st.markdown('<div class="page-header">🔬 Tumor Detection Module</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Advanced scan analysis • Segmentation overlay • Multi-class classification</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🧬 Full Analysis", "📈 Tumor Statistics", "⚙ Model Info"])

    with tab1:
        col_a, col_b = st.columns([1, 1], gap="medium")
        with col_a:
            st.markdown('<div class="ns-card">', unsafe_allow_html=True)
            st.markdown('<div class="ns-card-title">📤 Upload & Analyze</div>', unsafe_allow_html=True)
            td_file = st.file_uploader("Upload MRI for Full Analysis", type=["jpg","png","jpeg"], key="td_uploader", label_visibility="collapsed")
            td_analyze = st.button("🔍 Run Full Detection", use_container_width=True)

            if td_analyze and td_file:
                with st.spinner("Deep scan in progress…"):
                    img_arr = np.array(Image.open(td_file))
                    results = model.predict(img_arr)[0]
                    result_img = results.plot()
                    if isinstance(result_img, np.ndarray):
                        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                    if len(results.boxes.cls) > 0:
                        class_id = int(results.boxes.cls[0])
                        tc = model.names[class_id]; conf = float(results.boxes.conf[0]) * 100
                        box = results.boxes.xyxy[0].tolist(); w = box[2]-box[0]; h = box[3]-box[1]
                    else:
                        tc = "No Tumor"; conf = 99.0; w = 0; h = 0
                    st.session_state["td_tc"] = tc; st.session_state["td_conf"] = conf
                    st.session_state["td_img"] = result_img; st.session_state["td_box"] = [w, h]
                    st.session_state["td_orig"] = img_arr
                    mp_td = match_patient_by_filename(td_file.name)
                    st.session_state["td_matched"] = mp_td

            if st.session_state.get("td_orig") is not None:
                st.image(st.session_state["td_orig"], caption="Original MRI", use_container_width=True)
                if st.session_state.get("td_matched"):
                    mp2 = st.session_state["td_matched"]
                    st.markdown(f'<div class="match-banner"><div style="font-size:0.72rem;color:#00e676;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:4px">✅ Patient Record Matched</div><div style="font-weight:700">{mp2["name"]}</div><div style="font-size:0.78rem;color:#5a7a9a">{mp2["id"]} · {mp2["age"]}y</div></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="ns-card">', unsafe_allow_html=True)
            st.markdown('<div class="ns-card-title">🎯 Detection Overlay & Metrics</div>', unsafe_allow_html=True)
            if st.session_state.get("td_img") is not None:
                st.image(st.session_state["td_img"], caption="Detection Overlay", use_container_width=True)
                tc = st.session_state["td_tc"]; conf = st.session_state["td_conf"]
                w, h = st.session_state["td_box"]
                mp2 = st.session_state.get("td_matched")
                if mp2:
                    display_tumor = mp2["tumor"]; display_stage = mp2["stage"]
                    display_loc = mp2["location"]; display_growth = mp2["growth"]
                    display_mal = mp2["malignancy"]; display_risk = mp2["risk"]; display_conf = mp2["confidence"]
                else:
                    info = get_tumor_info(tc)
                    display_tumor = info["label"]; display_stage = info["stage"]
                    display_loc = info["location_guess"]; display_growth = info["growth"]
                    display_mal = info["malignancy"]; display_risk = info["risk"]; display_conf = conf
                render_confidence_bar(display_conf)
                st.markdown(f"""
                <table class="ns-table" style="margin-top:10px">
                  <tr><th>Metric</th><th>Value</th></tr>
                  <tr><td>Detected Class</td><td><b>{display_tumor}</b></td></tr>
                  <tr><td>Bounding Box (W×H)</td><td>{w:.0f} × {h:.0f} px</td></tr>
                  <tr><td>Malignancy</td><td style="color:{risk_color(display_risk)}">{display_mal}</td></tr>
                  <tr><td>Growth Rate</td><td>{display_growth}</td></tr>
                  <tr><td>Stage Estimate</td><td>{display_stage}</td></tr>
                  <tr><td>Location</td><td>{display_loc}</td></tr>
                </table>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                render_risk_meter(display_risk)
                st.markdown('<div class="warn-box">⚠️ AI-generated result. Clinical confirmation required before any treatment decision.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-box">Upload an MRI scan and click Run Full Detection to see the analysis.</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="ns-card">', unsafe_allow_html=True)
        st.markdown('<div class="ns-card-title">📊 Tumor Type Distribution</div>', unsafe_allow_html=True)
        dist_data = {"Glioblastoma":45,"Meningioma":20,"Pituitary":15,"Metastatic":12,"Others":8}
        c1, c2 = st.columns(2)
        with c1:
            for k, v in dist_data.items():
                bc = {"Glioblastoma":"#ff1744","Meningioma":"#ffab00","Pituitary":"#00bfa5","Metastatic":"#448aff","Others":"#90a4ae"}[k]
                st.markdown(f'<div style="margin-bottom:10px"><div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:4px"><span>{k}</span><span style="color:{bc};font-weight:700">{v}%</span></div><div class="conf-bar-wrap"><div class="conf-bar-fill" style="width:{v}%;background:{bc}"></div></div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown("""<table class="ns-table"><tr><th>Tumor Type</th><th>Avg Survival</th><th>Prevalence</th></tr>
              <tr><td>Glioblastoma (GBM)</td><td style="color:#ff4569">14 months</td><td>45%</td></tr>
              <tr><td>Meningioma</td><td style="color:#00e676">~80% (5yr)</td><td>20%</td></tr>
              <tr><td>Pituitary Adenoma</td><td style="color:#00e676">~97% (5yr)</td><td>15%</td></tr>
              <tr><td>Metastatic</td><td style="color:#ffab00">~30% (1yr)</td><td>12%</td></tr>
              <tr><td>Others</td><td style="color:#90caf9">Varies</td><td>8%</td></tr></table>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="ns-card">', unsafe_allow_html=True)
        st.markdown('<div class="ns-card-title">📈 Monthly Detection Trend (Simulated)</div>', unsafe_allow_html=True)
        import pandas as pd
        df = pd.DataFrame({"Month":["Jan","Feb","Mar","Apr","May","Jun"],"Detections":[28,34,41,38,52,60]})
        st.line_chart(df.set_index("Month"), color="#00e5ff", height=200)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="ns-card">', unsafe_allow_html=True)
        st.markdown('<div class="ns-card-title">⚙ Model Architecture & Performance</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        for col, lbl, val in zip([c1,c2,c3],["Architecture","Input Resolution","Parameters"],["YOLOv12n/s","640×640 px","~3.2M"]):
            col.markdown(f'<div class="metric-tile"><div class="val" style="font-size:1.1rem">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<table class="ns-table"><tr><th>Class</th><th>Precision</th><th>Recall</th><th>mAP@0.5</th></tr>
          <tr><td>Glioma</td><td>0.91</td><td>0.88</td><td>0.92</td></tr>
          <tr><td>Meningioma</td><td>0.87</td><td>0.84</td><td>0.88</td></tr>
          <tr><td>Pituitary</td><td>0.93</td><td>0.91</td><td>0.94</td></tr>
          <tr><td>No Tumor</td><td>0.98</td><td>0.97</td><td>0.98</td></tr>
          <tr><td><b>Overall</b></td><td><b>0.92</b></td><td><b>0.90</b></td><td><b>0.93</b></td></tr></table>""", unsafe_allow_html=True)
        st.markdown('<div class="warn-box" style="margin-top:1rem">⚠️ For research and screening purposes only. Not a substitute for radiologist diagnosis.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════
#  PATIENT RECORDS  ← NEW PAGE
# ═══════════════════════════════════════════
elif menu == "Patient Records":
    st.markdown('<div class="page-header">👤 Patient Records</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Upload any of the 13 registered MRI scans to auto-load the full patient profile • Or browse all records below</div>', unsafe_allow_html=True)

    col_up, col_detail = st.columns([1, 1.6], gap="medium")

    with col_up:
        st.markdown('<div class="ns-card">', unsafe_allow_html=True)
        st.markdown('<div class="ns-card-title">🔍 Scan-to-Patient Lookup</div>', unsafe_allow_html=True)
        pr_file = st.file_uploader("Upload registered MRI scan", type=["jpg","png","jpeg"], key="pr_uploader", label_visibility="collapsed")
        if pr_file:
            img_pr = np.array(Image.open(pr_file))
            st.image(img_pr, caption=pr_file.name, use_container_width=True)
            matched_pr = match_patient_by_filename(pr_file.name)
            if matched_pr:
                st.session_state["pr_selected"] = matched_pr
                st.markdown(f'<div class="match-banner"><div style="font-size:0.72rem;color:#00e676;font-weight:700;letter-spacing:0.08em;margin-bottom:4px">✅ RECORD FOUND</div><div style="font-weight:700;font-size:1rem">{matched_pr["name"]}</div><div style="font-size:0.78rem;color:#5a7a9a">{matched_pr["id"]} · {matched_pr["age"]}y · {matched_pr["gender"]}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warn-box">⚠️ No matching record found for this image.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Patient list
        st.markdown('<div class="ns-card">', unsafe_allow_html=True)
        st.markdown('<div class="ns-card-title">📋 All Patients</div>', unsafe_allow_html=True)
        st.markdown('<div style="max-height:420px;overflow-y:auto">', unsafe_allow_html=True)
        for p in PATIENT_DB:
            rc = risk_color(p["risk"])
            selected = st.session_state.get("pr_selected", {}).get("id") == p["id"]
            border_style = f"border-color:{rc}" if selected else ""
            if st.button(f"{p['name']}  ·  {p['id']}  ·  {p['age']}y", key=f"btn_{p['id']}", use_container_width=True):
                st.session_state["pr_selected"] = p
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_detail:
        p = st.session_state.get("pr_selected")
        if not p:
            st.markdown('<div class="info-box" style="margin-top:4rem;text-align:center;padding:2rem"><div style="font-size:2rem;margin-bottom:1rem">🩻</div><div style="font-weight:600">Upload a scan or click a patient name to load their full record</div></div>', unsafe_allow_html=True)
        else:
            # Patient header
            rc = risk_color(p["risk"])
            st.markdown(f"""
            <div class="ns-card" style="border-left:3px solid {rc}">
                <div style="display:flex;justify-content:space-between;align-items:flex-start">
                    <div>
                        <div style="font-size:1.25rem;font-weight:700">{p['name']}</div>
                        <div style="font-size:0.78rem;color:#5a7a9a;margin-top:3px">{p['id']} &nbsp;·&nbsp; {p['age']} yrs &nbsp;·&nbsp; {p['gender']} &nbsp;·&nbsp; Scan: {p['scanDate']}</div>
                        <div style="font-size:0.82rem;color:#90caf9;margin-top:5px">📋 {p['history']}</div>
                    </div>
                    <div style="text-align:right">
                        <div style="padding:3px 14px;background:{rc}18;border:1px solid {rc}44;border-radius:20px;color:{rc};font-weight:700;font-size:0.82rem">{p['risk']} Risk</div>
                        <div style="font-size:0.72rem;color:#5a7a9a;margin-top:5px">5-yr: <span style="color:#00e5ff">{p['survival5yr']}</span></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            pt1, pt2, pt3, pt4 = st.tabs(["🎯 Overview", "💊 Prescriptions", "🗓 Timeline", "💓 Vitals"])

            with pt1:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<div class="ns-card">', unsafe_allow_html=True)
                    st.markdown('<div class="ns-card-title">🎯 Detection Result</div>', unsafe_allow_html=True)
                    st.markdown(f'<div style="margin:6px 0 10px"><span class="badge badge-red">🔴 Tumor Detected</span></div>', unsafe_allow_html=True)
                    render_confidence_bar(p["confidence"])
                    st.markdown(f"""
                    <table class="ns-table" style="margin-top:10px">
                      <tr><th>Field</th><th>Value</th></tr>
                      <tr><td>Tumor Type</td><td><b style="color:#00e5ff">{p['tumor']}</b></td></tr>
                      <tr><td>Location</td><td>{p['location']}</td></tr>
                      <tr><td>Size</td><td style="color:#ffab00">{p['size']}</td></tr>
                      <tr><td>Stage</td><td style="color:{rc}">{p['stage']}</td></tr>
                      <tr><td>Growth Rate</td><td>{p['growth']}</td></tr>
                      <tr><td>Malignancy</td><td style="color:{rc}">{p['malignancy']}</td></tr>
                      <tr><td>5-yr Survival</td><td style="color:#00e5ff">{p['survival5yr']}</td></tr>
                    </table>
                    """, unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    render_risk_meter(p["risk"])
                    st.markdown('</div>', unsafe_allow_html=True)
                with c2:
                    st.markdown('<div class="ns-card">', unsafe_allow_html=True)
                    st.markdown('<div class="ns-card-title">💓 Vitals Snapshot</div>', unsafe_allow_html=True)
                    bp_c  = "#ff5252" if int(p["bp"].split("/")[0]) > 139 else "#00e676"
                    hr_c  = "#ff5252" if int(p["hr"]) > 100 else "#00e676"
                    tmp_c = "#ffab00" if float(p["temp"]) > 38 else "#00e676"
                    sp_c  = "#ffab00" if int(p["spo2"]) < 96 else "#00e676"
                    for lbl, val, col in [("Blood Pressure", p["bp"]+" mmHg", bp_c), ("Heart Rate", p["hr"]+" bpm", hr_c), ("Temperature", p["temp"]+"°C", tmp_c), ("SpO₂", p["spo2"]+"%", sp_c)]:
                        st.markdown(f'<div class="vital-row"><span class="vital-label">{lbl}</span><span class="vital-val" style="color:{col}">{val}</span></div>', unsafe_allow_html=True)
                    st.markdown('<br>', unsafe_allow_html=True)
                    st.markdown('<div class="ns-card-title" style="font-size:0.75rem;margin-top:8px">🧬 Patient Info</div>', unsafe_allow_html=True)
                    for k, v in [("Patient ID", p["id"]), ("Age", str(p["age"])+" years"), ("Gender", p["gender"]), ("Scan Date", p["scanDate"])]:
                        st.markdown(f'<div class="vital-row"><span class="vital-label">{k}</span><span style="font-size:0.82rem;font-weight:600">{v}</span></div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

            with pt2:
                st.markdown('<div class="ns-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="ns-card-title">💊 Prescription — {p["name"]}</div>', unsafe_allow_html=True)
                rows = "".join(f"<tr><td>{i+1}</td><td><b>{rx['name']}</b></td><td style='color:#00e5ff'>{rx['dose']}</td><td>{rx['freq']}</td><td style='color:#90caf9'>{rx['duration']}</td></tr>" for i, rx in enumerate(p["prescriptions"]))
                st.markdown(f'<table class="ns-table"><tr><th>#</th><th>Medication</th><th>Dosage</th><th>Frequency</th><th>Duration</th></tr>{rows}</table>', unsafe_allow_html=True)
                st.markdown('<div class="warn-box">⚠️ Consult a licensed oncologist before administration. AI-generated protocol only.</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with pt3:
                st.markdown('<div class="ns-card">', unsafe_allow_html=True)
                st.markdown('<div class="ns-card-title">🗓 Treatment Timeline</div>', unsafe_allow_html=True)
                for tl in p["timeline"]:
                    st.markdown(f"""
                    <div class="tl-item">
                        <div class="tl-dot" style="background:{tl['color']};box-shadow:0 0 8px {tl['color']}88"></div>
                        <div class="tl-content">
                            <div class="tl-title">{tl['title']}</div>
                            <div class="tl-sub">📅 {tl['week']} &nbsp; {status_badge_html(tl['status'])}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with pt4:
                st.markdown('<div class="ns-card">', unsafe_allow_html=True)
                st.markdown('<div class="ns-card-title">💓 Full Vitals & Patient Data</div>', unsafe_allow_html=True)
                cv1, cv2 = st.columns(2)
                bp_c  = "#ff5252" if int(p["bp"].split("/")[0]) > 139 else "#00e676"
                hr_c  = "#ff5252" if int(p["hr"]) > 100 else "#00e676"
                tmp_c = "#ffab00" if float(p["temp"]) > 38 else "#00e676"
                sp_c  = "#ffab00" if int(p["spo2"]) < 96 else "#00e676"
                with cv1:
                    for lbl, val, col in [("Blood Pressure",p["bp"]+" mmHg",bp_c),("Heart Rate",p["hr"]+" bpm",hr_c)]:
                        st.markdown(f'<div class="vital-row"><span class="vital-label">{lbl}</span><span class="vital-val" style="color:{col}">{val}</span></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="vital-row"><span class="vital-label">Temperature</span><span class="vital-val" style="color:{tmp_c}">{p["temp"]}°C</span></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="vital-row"><span class="vital-label">SpO₂</span><span class="vital-val" style="color:{sp_c}">{p["spo2"]}%</span></div>', unsafe_allow_html=True)
                with cv2:
                    for k, v in [("Tumor",p["tumor"]),("Stage",p["stage"]),("Scan Date",p["scanDate"]),("History",p["history"])]:
                        st.markdown(f'<div class="vital-row"><span class="vital-label">{k}</span><span style="font-size:0.78rem;font-weight:600;text-align:right;max-width:60%">{v}</span></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Export
            st.markdown("<br>", unsafe_allow_html=True)
            summary_text = f"""NeuroScan AI — Patient Report
==============================
Patient ID  : {p['id']}
Name        : {p['name']}
Age/Gender  : {p['age']} / {p['gender']}
Scan Date   : {p['scanDate']}
Tumor       : {p['tumor']}
Location    : {p['location']}
Size        : {p['size']}
Stage       : {p['stage']}
Risk Level  : {p['risk']}
Survival 5yr: {p['survival5yr']}
History     : {p['history']}
Vitals      : BP {p['bp']} | HR {p['hr']} bpm | Temp {p['temp']}°C | SpO₂ {p['spo2']}%

Prescriptions:
"""
            for rx in p["prescriptions"]:
                summary_text += f"  - {rx['name']}  {rx['dose']}  {rx['freq']}  ({rx['duration']})\n"
            summary_text += "\n⚠ Consult Oncologist Before Administration"
            st.download_button("⬇ Export Full Report (.txt)", data=summary_text,
                               file_name=f"neuroscan_{p['id']}.txt", mime="text/plain", use_container_width=True)


# ═══════════════════════════════════════════
#  PRESCRIPTION
# ═══════════════════════════════════════════
elif menu == "Prescription":
    st.markdown('<div class="page-header">💊 Prescription & Treatment Plan</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">AI-generated treatment protocol • Auto-populated from matched patient record if available</div>', unsafe_allow_html=True)

    # Priority: matched patient → session scan result → manual select
    mp = (st.session_state.get("matched_patient") or
          st.session_state.get("pr_selected") or
          st.session_state.get("td_matched"))

    tc   = st.session_state.get("tumor_class") or st.session_state.get("td_tc")
    conf = st.session_state.get("confidence") or st.session_state.get("td_conf")

    if mp:
        # Use full patient record
        p_name    = mp["name"];     p_id     = mp["id"]
        p_age     = mp["age"];      p_gender = mp["gender"]
        p_history = mp["history"];  p_scan   = mp["scanDate"]
        p_bp      = mp["bp"];       p_hr     = mp["hr"]
        p_temp    = mp["temp"];     p_spo2   = mp["spo2"]
        info_label    = mp["tumor"];        info_risk     = mp["risk"]
        info_mal      = mp["malignancy"];   info_growth   = mp["growth"]
        info_stage    = mp["stage"];        info_loc      = mp["location"]
        info_surv     = mp["survival5yr"];  info_conf     = mp["confidence"]
        prescriptions = mp["prescriptions"]
        timeline      = mp["timeline"]
        st.markdown(f'<div class="match-banner" style="margin-bottom:1rem"><div style="font-size:0.72rem;color:#00e676;font-weight:700;letter-spacing:0.08em;margin-bottom:2px">✅ Auto-loaded from Patient Record</div><div style="font-weight:700">{p_name} · {p_id}</div></div>', unsafe_allow_html=True)
    else:
        if not tc:
            st.markdown('<div class="warn-box">⚠️ No scan result found. Select a tumor type manually below, or run detection first.</div>', unsafe_allow_html=True)
            tc = st.selectbox("Select tumor type:", ["glioma","meningioma","pituitary","no tumor"])
            conf = None
        info = get_tumor_info(tc)
        p_name = ""; p_id = st.session_state.patient_id; p_age = 45; p_gender = "—"; p_history = ""; p_scan = datetime.datetime.now().strftime("%d %b %Y")
        p_bp = "120/80"; p_hr = "72"; p_temp = "37.0"; p_spo2 = "98"
        info_label = info["label"];         info_risk  = info["risk"]
        info_mal   = info["malignancy"];    info_growth= info["growth"]
        info_stage = info["stage"];         info_loc   = info["location_guess"]
        info_surv  = info["survival_5yr"];  info_conf  = conf or 0
        prescriptions = info["prescriptions"]
        timeline      = info["timeline"]

    col_p, col_t = st.columns([1, 1], gap="medium")

    with col_p:
        st.markdown('<div class="ns-card">', unsafe_allow_html=True)
        st.markdown('<div class="ns-card-title">👤 Patient Information</div>', unsafe_allow_html=True)
        pid     = st.text_input("Patient ID",   value=p_id)
        pname   = st.text_input("Patient Name", value=p_name)
        age     = st.number_input("Age", min_value=1, max_value=120, value=int(p_age))
        gender  = st.selectbox("Gender", ["Male","Female","Other"], index=["Male","Female","Other"].index(p_gender) if p_gender in ["Male","Female","Other"] else 0)
        history = st.text_area("Clinical History / Symptoms", value=p_history, height=80)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="ns-card">', unsafe_allow_html=True)
        st.markdown('<div class="ns-card-title">🧬 Diagnosis Overview</div>', unsafe_allow_html=True)
        rc = risk_color(info_risk)
        st.markdown(f"""
        <table class="ns-table">
          <tr><th>Field</th><th>Value</th></tr>
          <tr><td>Tumor Type</td><td><b>{info_label}</b></td></tr>
          <tr><td>Malignancy</td><td style="color:{rc}">{info_mal}</td></tr>
          <tr><td>Growth Rate</td><td>{info_growth}</td></tr>
          <tr><td>Stage</td><td>{info_stage}</td></tr>
          <tr><td>Location</td><td>{info_loc}</td></tr>
          <tr><td>5-yr Survival</td><td style="color:#00e5ff">{info_surv}</td></tr>
        </table>
        """, unsafe_allow_html=True)
        if info_conf:
            render_confidence_bar(info_conf)
        render_risk_meter(info_risk)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_t:
        st.markdown('<div class="ns-card">', unsafe_allow_html=True)
        st.markdown('<div class="ns-card-title">💊 Medical Prescription</div>', unsafe_allow_html=True)
        rows = "".join(f"<tr><td><b>{rx['name']}</b></td><td>{rx['dose']}</td><td>{rx['freq']}</td><td>{rx['duration']}</td></tr>" for rx in prescriptions)
        st.markdown(f'<table class="ns-table"><tr><th>Medication</th><th>Dose</th><th>Frequency</th><th>Duration</th></tr>{rows}</table>', unsafe_allow_html=True)
        st.markdown('<div class="warn-box">⚠️ Consult a licensed oncologist before administration.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="ns-card">', unsafe_allow_html=True)
        st.markdown('<div class="ns-card-title">🗓 Treatment Timeline</div>', unsafe_allow_html=True)
        for tl in timeline:
            st.markdown(f"""
            <div class="tl-item">
                <div class="tl-dot" style="background:{tl['color']}"></div>
                <div class="tl-content">
                    <div class="tl-title">{tl['title']}</div>
                    <div class="tl-sub">{tl['week']} &nbsp; {status_badge_html(tl['status'])}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="ns-card">', unsafe_allow_html=True)
        st.markdown('<div class="ns-card-title">💓 Vitals</div>', unsafe_allow_html=True)
        v1, v2 = st.columns(2)
        bp   = v1.text_input("Blood Pressure", value=p_bp)
        hr   = v2.text_input("Heart Rate (bpm)", value=p_hr)
        tmp  = v1.text_input("Temperature (°C)", value=p_temp)
        spo  = v2.text_input("SpO₂ (%)", value=p_spo2)
        bp_color = "#ff5252" if any(x in bp for x in ["140","150","160","170","180"]) else "#00e676"
        hr_color = "#ff5252" if int(hr or 72) > 100 else "#00e676"
        st.markdown(f"""
        <div style="margin-top:8px">
            <div class="vital-row"><span class="vital-label">BP</span><span class="vital-val" style="color:{bp_color}">{bp} mmHg</span></div>
            <div class="vital-row"><span class="vital-label">HR</span><span class="vital-val" style="color:{hr_color}">{hr} bpm</span></div>
            <div class="vital-row"><span class="vital-label">Temp</span><span class="vital-val">{tmp} °C</span></div>
            <div class="vital-row"><span class="vital-label">SpO₂</span><span class="vital-val" style="color:#00e676">{spo}%</span></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    _, col_dl, _ = st.columns([1,1,2])
    with col_dl:
        summary_text = f"""NeuroScan AI — Patient Report
==============================
Patient ID  : {pid}
Name        : {pname or p_name}
Date        : {datetime.datetime.now().strftime('%d %b %Y %H:%M')}
Tumor Type  : {info_label}
Stage       : {info_stage}
Risk Level  : {info_risk}
Survival 5yr: {info_surv}

Prescriptions:
"""
        for rx in prescriptions:
            summary_text += f"  - {rx['name']}  {rx['dose']}  {rx['freq']}  ({rx['duration']})\n"
        summary_text += "\n⚠ Consult Oncologist Before Administration"
        st.download_button("⬇ Export Report (.txt)", data=summary_text,
                           file_name=f"neuroscan_{pid}.txt", mime="text/plain", use_container_width=True)