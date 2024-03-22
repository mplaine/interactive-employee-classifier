FROM python:3.11.8-slim

# Change working directory
WORKDIR /usr/src/app

# Install curl, git, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone the project repository
# RUN git clone https://github.com/mplaine/interactive-employee-classifier.git .

# Copy the files required for dependencies to be installed
COPY requirements*.txt ./

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy all of the source code
COPY . .

# Expose port 8501
EXPOSE 8501

# Check the application's health on startup
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Start the Streamlit application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
