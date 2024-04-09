# Stage 1: Build stage
FROM python:3.11 as builder

# Set a working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --user -r requirements.txt

# Copy the rest of the application
COPY . .

# Stage 2: Run stage
FROM python:3.11-slim

# Create a non-root user
RUN useradd -m myuser

# Set the working directory
WORKDIR /app

# Copy only the dependencies installation from the 1st stage image
COPY --from=builder /root/.local /home/myuser/.local
COPY --from=builder /app /app

# Adjust permissions
RUN chmod -R 755 /app && chown -R myuser:myuser /app
USER myuser

# Ensure scripts in .local are usable:
ENV PATH=/home/myuser/.local/bin:$PATH

EXPOSE 3978

# Start the application
CMD ["python", "app.py"]