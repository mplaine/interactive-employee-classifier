FROM python:3.11.8-slim AS base

# Change working directory
WORKDIR /usr/src/app

# Install curl, add user "python", etc.
RUN apt-get update && \
    apt-get install -y curl && \
    groupadd -r python && useradd -g python python && \
    chown -R python:python . && \
    rm -rf /var/lib/apt/lists/*

# Copy the files required for dependencies to be installed
COPY --chown=python:python requirements*.txt ./

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy all of the source code
COPY --chown=python:python . .

# Switch to the "python" user
USER python

# Expose port 8501
EXPOSE 8501

# Check the application's health on startup
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Start the Streamlit application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
