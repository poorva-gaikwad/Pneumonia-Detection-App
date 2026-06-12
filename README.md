# 🫁 Pneuminia App

A production-grade, ML-powered web application for lung disease prediction, patient history management, and clinical analytics.

---

# 1. Project Overview

## What this project does

**Pneuminia App** is a full-stack medical web application that enables:

- Patient data collection
- Chest X-ray upload and analysis
- Machine learning–based disease prediction
- Historical record tracking
- Role-based dashboards for patients, doctors, and admins

It combines **clinical workflows + AI inference + analytics** in a single system.

---

## Problem it solves

Healthcare workflows often lack:

- Centralized patient history
- Fast preliminary scan analysis
- Easy-to-use ML-assisted tools

This system provides:

- A unified platform for patient + scan + prediction data
- Automated inference for lung diseases
- Long-term record storage for tracking health progression

---

## Key Features

- 🔐 Role-based authentication (`patient`, `doctor`, `admin`)
- 🧾 Patient history with symptoms, vitals, and risk factors
- 🩻 X-ray upload and validation
- 🤖 ML predictions:
  - Pneumonia (TensorFlow + YOLO)
  - Tuberculosis (TensorFlow)
  - Asthma (tabular ML model)

- 🔥 Grad-CAM heatmaps (model explainability)
- 📊 Analytics dashboard (trends, confidence, distributions)
- 🛡️ Security:
  - Rate limiting
  - reCAPTCHA
  - JWT authentication

- 🤖 AI assistant (Hugging Face / OpenAI-compatible)

---

## Target Users

- **Doctors** → clinical decision support
- **Patients** → record tracking & scan uploads
- **Admins** → system and user management
- **Developers** → ML web app reference architecture

---

# 2. Architecture Overview

## System Design

Monolithic Flask architecture with 4 layers:

```
[Client Browser]
        ↓
[Flask Routes (app1.py)]
        ↓
[Business Logic + Auth]
        ↓
 ├── Database (PostgreSQL)
 └── ML Models (TensorFlow / YOLO / Pickle)
```

---

## Data Flow

1. User logs in
2. Submits patient history
3. Uploads X-ray
4. Model processes image
5. Prediction stored in DB
6. Analytics generated from stored data

---

# 3. Tech Stack (Pinned Versions)

| Layer      | Technology         | Version |
| ---------- | ------------------ | ------- |
| Language   | Python             | 3.9.6   |
| Backend    | Flask              | 3.1.1   |
| ORM        | Flask-SQLAlchemy   | 3.1.1   |
| Auth       | Flask-Login        | 0.6.3   |
| JWT        | Flask-JWT-Extended | 4.7.1   |
| DB         | PostgreSQL         | 14+     |
| ML         | TensorFlow (CPU)   | 2.20.0  |
| Models     | Keras              | 3.10.0  |
| Detection  | YOLO (Ultralytics) | 8.3.217 |
| CV         | OpenCV             | 4.12.0  |
| Image      | Pillow             | 10.4.0  |
| Security   | Flask-Talisman     | 1.1.0   |
| Rate limit | Flask-Limiter      | 3.11.0  |

---

## Why These Choices (Critical)

- **Flask** → simplicity over scalability
- **PostgreSQL** → relational integrity + JSONB
- **TensorFlow** → compatibility with existing models
- **YOLO** → fast object detection
- **Monolith** → easier development, harder scaling

---

# 4. Setup & Installation

## Prerequisites

- Python 3.9.6
- PostgreSQL
- pip + venv

---

## Installation

```bash
python -m venv venv
source venv/bin/activate  # or Windows equivalent
pip install -r requirements.txt
```

If missing:

```bash
pip freeze > requirements.txt
```

---

## Environment Configuration

Create `.env`:

```env
APP_SECRET=your-secret
JWT_SECRET_KEY=your-jwt-secret
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/pneumonia_db
OPENAI_API_KEY=your-key
HF_TOKEN=your-token
ENABLE_CAPTCHA=false
```

---

## Run Application

```bash
python app1.py
```

Open:

```
http://127.0.0.1:5000
```

---

# 5. Database Design

## Core Tables

- `user`
- `patient_history`
- `asthma_record`
- `tb_record`
- `lung_cancer_record`
- `system_settings`

---

## Relationships

- User → PatientHistory (1:N)
- PatientHistory → Records (1:N)

---

## Migrations

```bash
flask db migrate -m "update"
flask db upgrade
```

---

# 6. Machine Learning Models (CRITICAL)

## Models Used

| Model           | File                    | Purpose            |
| --------------- | ----------------------- | ------------------ |
| Pneumonia Dense | modelDense.h5           | classification     |
| Pneumonia VGG16 | modelVGG16.h5           | classification     |
| TB Model        | best_model_epoch.keras  | TB detection       |
| YOLO            | pytorch_yolov8_model.pt | object detection   |
| Asthma          | model.pkl               | tabular prediction |

---

## Important Notes

- ⚠️ Models are **NOT versioned**
- ⚠️ Training code is **NOT included**
- ⚠️ Reproducibility is limited

---

## Input Assumptions

- Images resized (likely 224x224)
- Normalized pixel values
- Clean X-ray format expected

---

# 7. API Documentation

## Auth

- Session-based (browser)
- JWT (API)

---

## Example

### Login

```http
POST /api/login
```

```json
{
  "username": "user",
  "password": "pass"
}
```

---

### Get Current User

```http
GET /api/me
Authorization: Bearer <token>
```

---

# 8. Security Model

## Protections

- Rate limiting
- reCAPTCHA
- Secure headers (Talisman)
- Password hashing (scrypt)

---

## Risks (IMPORTANT)

- No antivirus scan for uploads
- Pickle model is unsafe
- No refresh tokens
- Limited file validation

---

# 9. Deployment

## Current State

- No Docker / CI/CD
- Runs via Flask dev server

---

## Recommended Production Setup

- Gunicorn / Waitress
- Nginx reverse proxy
- PostgreSQL (managed)
- Docker containerization

---

# 10. Backup & Recovery

## Backup Required

- Database
- `models/`
- `.env`
- `uploads/`

---

## Restore Steps

1. Restore DB
2. Replace models
3. Recreate env
4. Run migrations

---

# 11. Maintenance Guide

## Update Dependencies

```bash
pip install --upgrade <pkg>
pip freeze > requirements.txt
```

---

## Common Issues

| Issue            | Fix               |
| ---------------- | ----------------- |
| App not starting | Check `.env`      |
| Model crash      | Verify files      |
| DB error         | Check connection  |
| Upload fail      | Check permissions |

---

# 12. Known Limitations

- Monolithic architecture
- No automated tests
- No frontend framework
- Weak RBAC enforcement
- No audit logs

---

# 13. Future Improvements

- Modular architecture (Blueprints)
- Add Docker + CI/CD
- Add tests
- Replace pickle model
- Improve UI/UX
- Add API-first architecture

---

# 14. Environment Snapshot (TIME CAPSULE)

```
Year: 2026
Python: 3.9.6
Flask: 3.1.1
TensorFlow: 2.20.0
PostgreSQL: 14+
```

---

# 15. Quick Recovery Checklist

If system breaks:

- ✅ Check virtual environment
- ✅ Check `.env`
- ✅ Verify database
- ✅ Verify models exist
- ✅ Run migrations

---

# 16. Where to Start (For Future You)

1. `app1.py` → main logic
2. `models/` → ML files
3. `templates/` → UI
4. `migrations/` → DB history

---

# 17. License

This project is licensed under the MIT License.

You are free to:

Use
Modify
Distribute
Use commercially

See LICENSE file for details.
---
📩 Contact Me

If you need model files, collaboration, or support:

📧 Email: your-gaikwadpoorva7@gmail.com
🔗 GitHub: https://github.com/poorva-gaikwad
💬 Open an issue in this repository


