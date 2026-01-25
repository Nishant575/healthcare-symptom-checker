# Healthcare Symptom Checker - AI-Powered Medical Condition Classifier

An end-to-end machine learning project that predicts potential medical conditions based on patient symptoms. Built using PubMedBERT fine-tuned on the DDXPlus medical dataset with **99.73% accuracy** across 49 medical conditions.

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red.svg)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)
![Django](https://img.shields.io/badge/Django-5.0-green.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Model Validation](#model-validation)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training Details](#training-details)
- [API Endpoints](#api-endpoints)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Technologies Used](#technologies-used)
- [Future Work](#future-work)
- [License](#license)

---

## Overview

This project aims to assist in preliminary medical condition assessment by analyzing patient symptoms. The system uses a fine-tuned transformer model to classify symptoms into one of 49 medical conditions, providing healthcare professionals and patients with potential diagnoses for further investigation.

**Disclaimer:** This tool is for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.

### Features

- Fine-tuned PubMedBERT model for medical text understanding
- Multi-class classification across 49 medical conditions
- Class-weighted training for handling imbalanced data (246:1 ratio)
- Patient demographics (age, sex) included for improved accuracy
- Rigorous validation with data leakage detection and correction
- REST API backend with JWT authentication
- Production-ready model saved in HuggingFace format

---

## Key Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 99.73% |
| **F1 Score (Macro)** | 99.69% |
| **F1 Score (Weighted)** | 99.73% |
| **Training Time** | 9 hours 12 minutes |
| **Dataset Size** | 1,292,579 patient records |
| **Clean Test Samples** | 129,628 |

### Model Evolution: v1 → v2

| Aspect | v1 | v2 | Improvement |
|--------|----|----|-------------|
| Accuracy | 98.22% | 99.73% | +1.51% |
| F1 Macro | 97.94% | 99.69% | +1.75% |
| F1 Weighted | 98.10% | 99.73% | +1.63% |
| Train-Test Overlap | 99.35% | 3.64% | Fixed data leakage |
| Class Weights | No | Yes | Balanced rare classes |
| Symptoms Included | Max 12 | All | Richer features |
| Demographics | No | Yes | Age + Sex included |

### Difficult Classes Improvement

| Condition | v1 F1 | v2 F1 | Improvement |
|-----------|-------|-------|-------------|
| Acute rhinosinusitis | 53.91% | 92.18% | +71% |
| Chronic rhinosinusitis | 82.64% | 95.29% | +15% |
| Stable angina | 92.39% | 99.00% | +7% |
| Possible NSTEMI / STEMI | 93.92% | 100.00% | +6% |

### Rare Classes Performance (Class Weights Impact)

| Condition | Samples | F1 Score |
|-----------|---------|----------|
| Bronchiolitis | 36 | 100% |
| Ebola | 100 | 100% |
| Croup | 344 | 100% |
| Whooping cough | 549 | 100% |

---

## Model Validation

### Data Leakage Detection & Correction

During v1 development, validation revealed **99.35% overlap** between training and test sets due to limited symptom representation. This was corrected in v2 by:

1. Including ALL symptoms (not limited to 12)
2. Including symptom values (severity, location, duration)
3. Including patient demographics (age, sex)

### Validation Results

| Metric | All Test | Clean Test (no overlap) | Difference |
|--------|----------|-------------------------|------------|
| Accuracy | 99.74% | 99.73% | -0.01% |
| F1 Macro | 99.69% | 99.69% | -0.00% |
| F1 Weighted | 99.74% | 99.73% | -0.01% |

**Conclusion:** Negligible difference (-0.01%) confirms metrics are genuine and the model has truly learned to generalize.

---

## Dataset

This project uses the **DDXPlus Dataset**, a large-scale medical diagnosis dataset containing synthetic but realistic patient data.

### Why Synthetic Data?

Real medical data (MIMIC-IV, UK Biobank) requires credentialed access and data use agreements. DDXPlus was chosen because:
- Medically validated by healthcare professionals
- Published in peer-reviewed research
- 1.3M records with ground truth labels
- Publicly deployable without legal restrictions
- Demonstrates identical ML engineering skills

| Split | Records |
|-------|---------|
| Training | 1,025,602 |
| Validation | 132,448 |
| Test | 134,529 |
| **Total** | **1,292,579** |

### Dataset Characteristics

- **49 Medical Conditions** including respiratory, cardiovascular, neurological, and infectious diseases
- **223 Unique Symptoms** covering various body systems
- **Balanced Demographics:** 51.5% Female, 48.5% Male
- **Age Range:** 0-109 years (Mean: 39.7 years)
- **Class Imbalance:** 246:1 ratio (URTI: 64,368 vs Bronchiolitis: 261)

### Conditions Covered

<details>
<summary>Click to expand full list of 49 conditions</summary>

1. Acute COPD exacerbation / infection
2. Acute dystonic reactions
3. Acute laryngitis
4. Acute otitis media
5. Acute pulmonary edema
6. Acute rhinosinusitis
7. Allergic sinusitis
8. Anaphylaxis
9. Anemia
10. Atrial fibrillation
11. Boerhaave
12. Bronchiectasis
13. Bronchiolitis
14. Bronchitis
15. Bronchospasm / acute asthma exacerbation
16. Chagas
17. Chronic rhinosinusitis
18. Cluster headache
19. Croup
20. Ebola
21. Epiglottitis
22. GERD
23. Guillain-Barré syndrome
24. HIV (initial infection)
25. Influenza
26. Inguinal hernia
27. Larygospasm
28. Localized edema
29. Myasthenia gravis
30. Myocarditis
31. PSVT
32. Pancreatic neoplasm
33. Panic attack
34. Pericarditis
35. Pneumonia
36. Possible NSTEMI / STEMI
37. Pulmonary embolism
38. Pulmonary neoplasm
39. SLE
40. Sarcoidosis
41. Scombroid food poisoning
42. Spontaneous pneumothorax
43. Spontaneous rib fracture
44. Stable angina
45. Tuberculosis
46. URTI
47. Unstable angina
48. Viral pharyngitis
49. Whooping cough

</details>

---

## Project Structure

```
healthcare-symptom-checker/
│
├── notebooks/
│   ├── 01_data_preparation.ipynb       # v1: Initial EDA and preprocessing
│   ├── 02_model_training.ipynb         # v1: Initial model training
│   ├── 03_model_validation.ipynb       # Data leakage detection
│   ├── 01_data_preparation_v2.ipynb    # v2: Improved preprocessing
│   └── 02_model_training_v2.ipynb      # v2: Training with class weights
│
├── data/
│   ├── ddxplus/                        # Raw DDXPlus dataset
│   │   ├── release_conditions.json
│   │   ├── release_evidences.json
│   │   ├── release_train_patients.zip
│   │   ├── release_validate_patients.zip
│   │   └── release_test_patients.zip
│   │
│   ├── classifier/                     # v1 processed data
│   │   └── ...
│   │
│   └── classifier_v2/                  # v2 processed data
│       ├── train.json                  # Training data (983 MB)
│       ├── val.json                    # Validation data (129 MB)
│       ├── test.json                   # Test data (131 MB)
│       ├── label2id.json
│       └── id2label.json
│
├── models/
│   └── condition_classifier_v2/
│       └── final/                      # Trained model
│           ├── model.safetensors
│           ├── config.json
│           ├── tokenizer.json
│           ├── vocab.txt
│           ├── label2id.json
│           ├── id2label.json
│           ├── training_metrics.json
│           └── classification_report.txt
│
├── backend/
│   ├── config/                         # Django settings
│   │   ├── settings.py
│   │   └── urls.py
│   ├── users/                          # User authentication
│   │   ├── models.py
│   │   ├── serializers.py
│   │   ├── views.py
│   │   └── urls.py
│   ├── predictor/                      # ML prediction service
│   │   ├── models.py
│   │   ├── ml_service.py
│   │   ├── serializers.py
│   │   ├── views.py
│   │   └── urls.py
│   ├── manage.py
│   └── .env
│
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (recommended)
- 16GB+ RAM
- 15GB+ free disk space

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Nishant575/healthcare-symptom-checker.git
cd healthcare-symptom-checker
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download DDXPlus dataset**

   Download from [DDXPlus on Figshare](https://figshare.com/articles/dataset/DDXPlus_Dataset/20043374) and place in `data/ddxplus/`:
   - `release_conditions.json`
   - `release_evidences.json`
   - `release_train_patients.zip`
   - `release_validate_patients.zip`
   - `release_test_patients.zip`

5. **Setup Backend**
```bash
cd backend
python manage.py migrate
python manage.py createsuperuser
```

---

## Usage

### Running the Notebooks

1. **Start Jupyter Notebook**
```bash
jupyter notebook
```

2. **Run notebooks in order:**
   - `01_data_preparation_v2.ipynb` - Data loading and preprocessing
   - `02_model_training_v2.ipynb` - Model training with class weights

### Using the Trained Model

```python
from transformers import pipeline

# Load the model
classifier = pipeline(
    "text-classification",
    model="models/condition_classifier_v2/final",
    tokenizer="models/condition_classifier_v2/final",
    top_k=5
)

# Make predictions (include age and sex for best results)
text = "Patient is a 35 year old Male presenting with: runny nose; sore throat; cough; fever"
results = classifier(text)[0]

for result in results:
    print(f"{result['label']}: {result['score']*100:.2f}%")
```

### Running the API

```bash
cd backend
python manage.py runserver
```

API will be available at `http://127.0.0.1:8000/`

---

## Model Architecture

### Base Model

**PubMedBERT** (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`)

- Pre-trained on PubMed abstracts (medical/biomedical text)
- BERT architecture optimized for medical domain
- 109.5M parameters

### Fine-tuning Configuration

| Parameter | Value |
|-----------|-------|
| Max Sequence Length | 512 tokens |
| Batch Size | 4 (effective: 64 with gradient accumulation) |
| Gradient Accumulation | 16 steps |
| Learning Rate | 2e-5 |
| Weight Decay | 0.01 |
| Warmup Ratio | 0.1 |
| Epochs | 1 |
| Optimizer | AdamW |
| Precision | FP16 (mixed precision) |
| Class Weights | Balanced (sklearn) |
| Best Metric | F1 Macro |

### Class Weights

To handle the 246:1 class imbalance:

| Class Type | Weight Range | Example |
|------------|--------------|---------|
| Rare classes | 3.5 - 80.2 | Bronchiolitis: 80.19 |
| Common classes | 0.3 - 0.8 | URTI: 0.33 |

---

## Training Details

### Hardware

- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM)
- **Training Time:** 9 hours 12 minutes (1 epoch)

### Input Text Format

```
Patient is a 35 year old Male presenting with: has fever; 
pain location: forehead; pain intensity: 4/10; has cough; 
pain characterized as: sharp; has traveled recently: N
```

### Key Training Decisions

| Decision | Rationale |
|----------|-----------|
| 1 Epoch | v1 showed epochs 2-3 only added +0.06% accuracy |
| Class Weights | Required for 246:1 imbalance ratio |
| MAX_LENGTH=512 | Longer texts need more tokens |
| F1 Macro metric | Better for imbalanced classification |

---

## API Endpoints

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/api/auth/register/` | User registration | No |
| POST | `/api/auth/login/` | Login, get JWT token | No |
| GET | `/api/auth/profile/` | Get user profile | Yes |
| POST | `/api/predict/` | Get condition prediction | Yes |
| GET | `/api/history/` | Get prediction history | Yes |
| GET | `/api/conditions/names/` | List all 49 conditions | No |

### Example: Make Prediction

```bash
# Login
curl -X POST http://127.0.0.1:8000/api/auth/login/ \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password123"}'

# Predict (use token from login response)
curl -X POST http://127.0.0.1:8000/api/predict/ \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"symptoms": "runny nose; sore throat; cough; mild fever"}'
```

---

## Exploratory Data Analysis

### Key Findings

1. **Class Distribution:** URTI most common (6.28%), Bronchiolitis least common (0.03%)
2. **Age Distribution:** Mean 39.7 years, covers infants to elderly
3. **Sex Distribution:** Nearly balanced (51.5% Female, 48.5% Male)
4. **Symptoms per Patient:** Average of 20 symptoms per record
5. **Class Imbalance:** 246:1 ratio addressed with class weights

### Visualizations

#### Condition Distribution
![Condition Distribution](notebooks/images/condition_distribution.png)

#### Age Distribution
![Age Distribution](notebooks/images/age_distribution.png)

#### Age by Condition
![Age by Condition](notebooks/images/age_by_condition.png)

---

## Technologies Used

| Category | Technologies |
|----------|--------------|
| **Language** | Python 3.13 |
| **Deep Learning** | PyTorch 2.6.0, HuggingFace Transformers |
| **Data Processing** | Pandas, NumPy |
| **ML Utilities** | scikit-learn, datasets |
| **Backend** | Django 5.0, Django REST Framework |
| **Authentication** | JWT (djangorestframework-simplejwt) |
| **Visualization** | Matplotlib |
| **Development** | Jupyter Notebook |
| **Hardware** | NVIDIA RTX 4050 (CUDA 12.4) |

---

## Future Work

- [ ] **Part 3: Frontend** - React-based user interface
- [ ] **Part 4: Deployment** - Docker containerization and cloud deployment
- [ ] Add NER model for symptom extraction from free text
- [ ] Implement urgency classification
- [ ] Add model explainability (attention visualization)
- [ ] Expand condition coverage

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **DDXPlus Dataset:** [Fansi Tchango et al.](https://figshare.com/articles/dataset/DDXPlus_Dataset/20043374)
- **PubMedBERT:** [Microsoft Research](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract)
- **HuggingFace:** For the Transformers library

---

## Contact

**Nishant Dalvi**
- GitHub: [@Nishant575](https://github.com/Nishant575)
- LinkedIn: [Nishant Dalvi](https://linkedin.com/in/nishant-dalvi)
- Email: ndalvi.cs@gmail.com

---

*If you find this project useful, please consider giving it a ⭐!*