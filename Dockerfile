FROM --platform=linux/amd64 ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install --no-install-recommends -y -q \
        apt-utils \
        build-essential \
        ca-certificates \
        curl \
        git \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools \
        wget \
        zip

####################
# Application dependencies
RUN apt-get update
RUN pip3 install --upgrade pip
COPY requirements.dev.txt /tmp/
COPY requirements.txt /tmp/
RUN pip3 install --requirement /tmp/requirements.dev.txt
RUN pip3 install --requirement /tmp/requirements.txt

####################
# Add Jupyter notebook
# Tini operates as a process subreaper for jupyter, preventing kernel crashes.
# Pickleshare is required for some magic methods in jupyter.
WORKDIR /src/
RUN pip3 install jupyter pickleshare
ADD https://github.com/krallin/tini/releases/download/v0.6.0/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
EXPOSE 8888
ENTRYPOINT ["/usr/bin/tini", "--"]