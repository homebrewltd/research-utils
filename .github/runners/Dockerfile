FROM python:3.11-slim

# Install necessary packages
RUN apt-get update && apt-get install -y \
    curl \
    tar \
    unzip \
    git \
    jq \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
  unzip awscliv2.zip && \
  ./aws/install

ENV HOME=/home/runner

RUN mkdir -p /home/runner

ARG RUNNER_VERSION=2.317.0

ARG RUNNER_UID=1000
ARG DOCKER_GID=1001

RUN adduser --disabled-password --gecos "" --uid $RUNNER_UID runner \
    && groupadd docker --gid $DOCKER_GID \
    && usermod -aG sudo runner \
    && usermod -aG docker runner \
    && echo "%sudo   ALL=(ALL:ALL) NOPASSWD:ALL" > /etc/sudoers \
    && echo "Defaults env_keep += \"DEBIAN_FRONTEND\"" >> /etc/sudoers

# cd into the user directory, download and unzip the github actions runner
RUN cd /home/runner && mkdir actions-runner && cd actions-runner \
    && curl -O -L https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz \
    && tar xzf ./actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz

RUN chown -R runner:runner /home/runner && /home/runner/actions-runner/bin/installdependencies.sh

ADD ./start.sh /home/runner/start.sh

RUN chmod +x /home/runner/start.sh

ENTRYPOINT ["/bin/bash", "/home/runner/start.sh"]

USER runner