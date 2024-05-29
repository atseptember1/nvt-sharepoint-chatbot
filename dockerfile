# Specify the source image
FROM python:3.11-slim

# Set a working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --user -r requirements.txt

# Copy the rest of the application
COPY . .

# Ensure scripts in .local are usable:
ENV PATH=/home/myuser/.local/bin:$PATH

EXPOSE 3978

# Start the application
CMD ["python", "app.py"]
