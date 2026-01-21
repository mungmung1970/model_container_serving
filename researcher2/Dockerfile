FROM jupyter/scipy-notebook:python-3.10

WORKDIR /home/jovyan/work

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

VOLUME ["/artifacts"]
