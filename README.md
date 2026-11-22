# Kerala Seasonal Bird Diversity – Streamlit demo

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Deploy free on Streamlit Community Cloud
1. Create a GitHub repo and upload everything in this folder (total ~5 MB, within GitHub limits).
2. Go to share.streamlit.io -> New app -> pick the repo, main file `app.py`.
3. Deploy; you get a public link.

`requirements.txt` pins scikit-learn==1.6.1 because `best_model.joblib` was trained with that version.
Python 3.11 or 3.12 recommended (set under Advanced settings on Streamlit Cloud).
