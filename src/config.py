from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

JD_CACHE_PATH = PROCESSED_DIR / "jd_cache.json"
CV_CACHE_PATH = PROCESSED_DIR / "cv_cache.json"
EXTRACTED_DATASET_PATH = PROCESSED_DIR / "extracted_dataset.json"

# Domain Constants
LABEL_MAP = {
    "No Fit": 0,
    "Potential Fit": 1,
    "Good Fit": 2,
}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

MAJOR_GROUPS = [
    "computer_science",
    "information_technology",
    "software_engineering",
    "data_science_ai",
    "electrical_electronics",
    "mechanical_engineering",
    "civil_construction",
    "accounting_finance",
    "business_administration",
    "marketing_communications",
    "human_resources",
    "economics",
    "healthcare_medical",
    "biology_biotechnology",
    "chemistry_chemical_engineering",
    "physics_materials_science",
    "mathematics_statistics",
    "law_legal",
    "languages_linguistics",
    "graphic_design_multimedia",
    "supply_chain_logistics",
    "hospitality_tourism",
    "education_pedagogy",
    "environmental_science",
    "general_engineering",
    "other",
    "unknown",
]

CV_HEADERS = [
    "Professional Summary", "Summary", "Executive Profile", "Profile", "Objective",
    "Highlights", "Skills", "Core Qualifications", "Qualifications", "Technical Skills",
    "Experience", "Work History", "Professional Experience", "Education and Training",
    "Education", "Certifications", "Certification", "Accomplishments", "Additional Information",
    "Languages", "Interests", "Affiliations", "Training",
]

# Defaults
DEFAULT_DATASET_NAME = "cnamuangtoun/resume-job-description-fit"
DEFAULT_VLLM_MODEL = "Qwen/Qwen3-14B-FP8"
DEFAULT_RANDOM_STATE = 42
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15
