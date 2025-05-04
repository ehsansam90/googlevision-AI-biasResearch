# Safe Search Detection with Google Cloud Vision API

This project analyzes the safety of images using Google Cloud's Vision API, specifically leveraging the Safe Search Detection feature. It processes images from categorized researcher folders and outputs CSV files containing safety labels for each image.

## 📦 Features

- Uses Google Cloud Vision's Safe Search API to detect:
  - Adult content
  - Medical content
  - Spoof (fake content)
  - Violence
  - Racy content
- Processes multiple researcher folders and images in batch
- Outputs results to CSV format

## 🛠️ Requirements

- Python 3.6+
- Google Cloud Vision API enabled
- A Google Cloud service account key file (`.json`)
- Internet connection (to call the API)

## 📁 Project Structure

project/
│
├── config.ini
├── your_script.py
├── Pilot Sample/
│ └── [Researcher Folders]/
│ └── Additional Images/
├── PilotResults_SafeSearchDetection_additionImages/
│ └── [Results CSVs]
└── visionai-xxxxxx-yyyyyyyyyyyy.json


## ⚙️ Setup

1. **Install Dependencies**
   ```bash
   pip install google-cloud-vision pandas

[google]
credentials_path = visionai-xxxxxx-yyyyyyyyyyyy.json
