FROM python:3.9

COPY ./services/orchestrator_proxy/requirements.txt /tmp/requirements.app.txt
#COPY ./libs/requirements.txt /tmp/requirements.libs.txt
RUN pip install --no-cache-dir --upgrade -r /tmp/requirements.app.txt
#RUN pip install --no-cache-dir --upgrade -r /tmp/requirements.libs.txt

WORKDIR /src
