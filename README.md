# 🎓 Student Performance Predictor

An interactive **Streamlit + Machine Learning** web application that estimates whether a student is likely to **Pass** or **Fail** using study habits and learning-resource indicators.

**Created by Ajim Sikandar Mujawar** · Data Science & Artificial Intelligence

## ✨ Features

- Clean, responsive dashboard with a student-friendly interface
- Inputs for weekly study hours, attendance, previous exam score, extracurricular activities, and home internet access
- Predictions from a pre-trained Random Forest classifier
- Probability breakdown for each predicted class
- Input summary for reviewing a prediction
- Model loaded once with Streamlit caching
- Clear educational-use disclaimer

## 🖥️ Preview

The app opens with a dashboard header, student profile form, and prediction results section.

## 🗂️ Project structure

```text
student-result-predictor/
├── app.py              # Streamlit user interface and prediction workflow
├── model.pkl           # Pre-trained model, encoders, and feature metadata
├── requirements.txt    # Python dependencies
├── .gitignore          # Files Git should ignore
├── LICENSE             # MIT license
└── README.md           # Project documentation
```

## 🚀 Run locally

### 1. Clone the repository

```bash
git clone https://github.com/ajimmujawar44/student-performance-predictor.git
cd student-performance-predictor
```



### 2. Create and activate a virtual environment

**Windows PowerShell**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the app

```bash
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`. Open it in your browser.

## 🧠 Model inputs

| Feature | Description | Expected range/options |
|---|---|---|
| Study hours per week | Approximate weekly study time | 0–100 hours |
| Attendance | Percentage of classes attended | 0–100% |
| Previous exam score | Score from a previous exam | 0–100 |
| Extracurricular activities | Whether the student participates | No / Yes |
| Internet access at home | Whether home internet is available | No / Yes |

The app uses the encoders bundled with `model.pkl` for categorical features, preserving the model's expected input representation.

## 🧰 Tech stack

- **Python** — application logic
- **Streamlit** — interactive web interface
- **Pandas** — input data preparation
- **Scikit-learn** — machine-learning classifier and preprocessing artifacts
- **Joblib** — loading the serialized model bundle

## ☁️ Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Select **Create app** and choose your repository, branch, and `app.py` as the main file.
4. Deploy. Keep `model.pkl` in the repository root, beside `app.py`.

## 📤 Upload to GitHub

From the project directory:

```bash
git init
git add app.py model.pkl requirements.txt README.md .gitignore LICENSE
git commit -m "Build polished student performance predictor app"
git branch -M main
git remote add origin https://github.com/ajimmujawar44/student-performance-predictor.git
git push -u origin main
```

If Git is already initialized, do not run `git init` again; add the files, commit, and push to the existing remote instead.

## ⚠️ Important limitations

- This project is an educational demonstration, not a validated academic assessment tool.
- Predictions depend on the training dataset and model quality. A probability is not a guarantee.
- Avoid using this app to make consequential decisions about real students without rigorous validation, fairness checks, privacy safeguards, and appropriate human oversight.
- Only publish data and artifacts that you have permission to share.

## 👨‍💻 Author

**Ajim Sikandar Mujawar**  
Aspiring Data Scientist | Machine Learning | AI

---

If you find this project useful, consider giving the repository a ⭐ on GitHub.
