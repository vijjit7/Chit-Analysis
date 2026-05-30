FROM python:3.12-slim

# System dependency: Tesseract OCR engine, required by pytesseract for the
# screenshot-upload (OCR) feature. Without it the OCR calls fail at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so they cache across code-only changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway injects $PORT=8080 at runtime and routes its public domain to the
# EXPOSEd port, so EXPOSE must match the runtime port or the proxy targets the
# wrong port ("Application failed to respond"). 8080 is also the default for
# local `docker run`.
ENV PORT=8080
EXPOSE 8080

# Shell form so $PORT expands at container start.
CMD streamlit run app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
