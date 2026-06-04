FROM python:3.12-slim
WORKDIR /usr/local/app

# Install the application dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

# Copy in the source code
COPY api.py preprocessing.py train.py CNN.py ./
COPY models ./models
EXPOSE 8000

# Setup an app user so the container doesn't run as the root user
RUN useradd app
USER app

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

