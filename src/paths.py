from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "playground-series-s6e4"
TRAIN_CSV = DATA / "train.csv"
TEST_CSV = DATA / "test.csv"
SAMPLE_SUB_CSV = DATA / "sample_submission.csv"
SUBMISSIONS = ROOT / "submissions"

TARGET = "Irrigation_Need"
CLASSES = ["Low", "Medium", "High"]

CATEGORICAL = [
    "Soil_Type",
    "Crop_Type",
    "Crop_Growth_Stage",
    "Season",
    "Irrigation_Type",
    "Water_Source",
    "Mulching_Used",
    "Region",
]
NUMERIC = [
    "Soil_pH",
    "Soil_Moisture",
    "Organic_Carbon",
    "Electrical_Conductivity",
    "Temperature_C",
    "Humidity",
    "Rainfall_mm",
    "Sunlight_Hours",
    "Wind_Speed_kmh",
    "Field_Area_hectare",
    "Previous_Irrigation_mm",
]
