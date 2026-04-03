# Use a lightweight Python image
FROM python:3.11-slim

# Create a non-root user (mandatory for Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY --chown=user . .

# Expose the port Hugging Face expects
EXPOSE 7860

# Start the FastAPI server from the canonical OpenEnv entry point
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
