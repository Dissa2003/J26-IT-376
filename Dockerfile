FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shared_contracts ./shared_contracts
ARG MODULE
COPY ${MODULE} ./${MODULE}
ENV MODULE=${MODULE} PYTHONUNBUFFERED=1
CMD python -m ${MODULE}