FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

> ⚠️ Hugging Face Spaces uses **port 7860** — this is important!

---

### STEP 4: Update `requirements.txt`
```
fastapi==0.111.0
uvicorn==0.30.1
pydantic==2.7.1
numpy==1.26.4
scikit-learn==1.4.2
joblib==1.4.2