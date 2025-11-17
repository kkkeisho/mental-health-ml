# Mental Health ML

This project turns a mental-health-in-tech survey into a helpful tool:
we clean the answers, train a machine-learning model, and offer a small
website-like service that can tell whether someone is likely to seek
treatment.

You can use it even if you are new to coding—everything below walks you
through the steps slowly.

---

## 1. What You Need

- A computer with **Python 3.9+**. If you are unsure, open a terminal
  (macOS/Linux) or Command Prompt (Windows) and type `python --version`.
- An internet connection to download this folder and install the few
  tools we use.
- About 10–15 minutes for the initial setup.

---

## 2. Quick Start (copy & paste)

Open your terminal/command prompt, go to the place where you want the
project folder, then run the following block **line by line**:

```bash
git clone <repo-url> mental-health-ml   # download this project
cd mental-health-ml                     # move into the folder
python -m venv .venv                    # create a private Python space
source .venv/bin/activate               # turn it on (Windows: .venv\Scripts\activate)
pip install --upgrade pip               # make sure pip is recent
pip install -r requirements.txt         # install everything we need
```

What just happened?

- `git clone` copies the project to your computer.
- `python -m venv .venv` creates an isolated box so that the libraries we
  install do not interfere with other projects.
- `source .venv/bin/activate` (or `.venv\Scripts\activate` on Windows)
  turns that box on. You will see `(.venv)` at the start of your prompt.
- The two `pip` commands install the right packages (FastAPI,
  scikit-learn, etc.).

That is the entire setup.

---

## 3. How the Pieces Fit Together

| Folder / File | Plain-English meaning |
| ------------- | --------------------- |
| `data/` | The survey answers we start from (`survey.csv`). |
| `src/preprocess.py` | Cleans the survey (fix ages, tidy genders, handle missing values). |
| `src/train.py` | Teaches the computer model using the cleaned survey. |
| `models/model.pkl` | The saved model created by the training step. |
| `api/app.py` | A tiny web service that loads the model and answers prediction requests. |
| `api/schemas.py` | Defines what a “valid survey submission” looks like. |
| `tests/` | Automated checks to confirm the app responds correctly. |
| `mlruns/` | A diary written by MLflow so you can review experiments later. |

---

## 4. Typical Workflow (Step by Step)

1. **(Optional) Look at the data**
   - Open `notebooks/eda.ipynb` in Jupyter (or VS Code) if you want to
     see charts and summaries.

2. **Train / refresh the model**
   - Run:
     ```bash
     python -m src.train --do-split
     ```
   - This reads `data/survey.csv`, cleans it, trains a logistic
     regression model, evaluates it on a test split, and stores the
     result in `models/model.pkl`.
   - You will see accuracy/precision/recall numbers printed at the end.

3. **(Optional) Review experiments in a browser**
   ```bash
   mlflow ui --backend-store-uri mlruns
   ```
   - Open http://127.0.0.1:5000 and compare training runs (handy when
     you start experimenting with different settings).

4. **Serve predictions (like a small API)**
   - Start the server:
     ```bash
     uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
     ```
   - Leave this command running. It prints log messages to let you know
     it is alive.
   - Visit http://localhost:8000/docs to see a friendly interface where
     you can try the `/predict` endpoint without writing code. Fill the
     form and click “Execute” to receive a prediction plus the model’s
     confidence score.

5. **Run the safety checks (tests)**
   ```bash
   pytest
   ```
   - Confirms the API still behaves as expected.

---

## 5. Making a Prediction Manually (optional)

If you prefer using the command line instead of the interactive docs,
send a request with `curl` once the server is running:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "Age": 30,
        "Gender": "Male",
        "Country": "Japan",
        "self_employed": "No",
        "family_history": "No",
        "work_interfere": "Sometimes",
        "no_employees": "6-25",
        "remote_work": "Yes",
        "tech_company": "Yes",
        "benefits": "Yes",
        "care_options": "Not sure",
        "wellness_program": "No",
        "seek_help": "Yes",
        "anonymity": "Yes",
        "leave": "Somewhat easy",
        "mental_health_consequence": "No",
        "phys_health_consequence": "No",
        "coworkers": "Some of them",
        "supervisor": "Yes",
        "mental_health_interview": "No",
        "phys_health_interview": "Yes",
        "mental_vs_physical": "Don't know",
        "obs_consequence": "No"
      }'
```

The response looks like:

```json
{
  "prediction": 1,
  "probability_yes": 0.78,
  "model": "logistic-regression + preprocessing pipeline"
}
```

Where `prediction: 1` means “likely to seek treatment” and `0` would mean
“unlikely.”

---

## 6. Need Ideas for Next Steps?

- Replace `data/survey.csv` with your own survey in the same format,
  re-run the training step, and the API will use the new model.
- Adjust the training command to try different parameters (for example
  `--test-size 0.3` to hold out 30% of the data for testing).
- Share the running API with teammates using tools such as
  [ngrok](https://ngrok.com/).

Whenever you change the data cleaning or the survey structure, remember
to:
1. Re-run `python -m src.train --do-split`
2. Restart the `uvicorn` server so it loads the new `models/model.pkl`

---

## 7. Troubleshooting

- **Command not found**: make sure your virtual environment is activated
  (you should see `(.venv)` near the start of the prompt). If not, run
  `source .venv/bin/activate` (macOS/Linux) or `.venv\Scripts\activate`
  (Windows PowerShell/CMD).
- **ModuleNotFoundError**: run `pip install -r requirements.txt` again.
- **Port already in use** when starting the API: another program might be
  using port 8000. Either stop it or run
  `uvicorn api.app:app --port 9000`.

You can always stop a running command by pressing `Ctrl + C`.

---

## 8. Summary

1. Install dependencies (one-time).
2. Train the model with `python -m src.train --do-split`.
3. Serve predictions with `uvicorn api.app:app --reload`.
4. Visit http://localhost:8000/docs to interact with the model—no coding
   required.

Enjoy exploring the mental-health survey and feel free to customize the
pipeline as you grow more comfortable with the tools!

