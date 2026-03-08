# Python Environment
FROM python:3.11-slim 

#Working directory inside container where all commandas run
WORKDIR /app


COPY requirements.txt .

# install required liabraries
RUN pip install --no-cache-dir -r requirements.txt

# copy the code of the project
COPY . .

# for API open port 8000 and for streamlit open port 8501
EXPOSE 8000
EXPOSE 8501

# Start the FastAPI
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]