FROM python:3.10

# Create the working directory
RUN set -ex && mkdir /translation
WORKDIR /translation

# Install Python dependencies
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Copy the relevant directories
COPY model/ ./model
COPY . ./

# Run the web server
EXPOSE 7860
ENV PYTHONPATH /translation
CMD python /translation/app.py 

