# Prophet needs a C++ toolchain (it compiles a Stan model on install),
# so we start from the full python image rather than slim.
FROM python:3.11

WORKDIR /app

# Install dependencies first so Docker can cache this layer when only
# the application code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project.
COPY . .

# Build the processed tables and train the models at image-build time so
# the container starts ready to serve. (Comment these out if you'd rather
# mount pre-trained models from the host.)
RUN python src/data_pipeline.py && python src/train.py

EXPOSE 8000 7860

# Default command runs the API. docker-compose overrides this for the UI.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
