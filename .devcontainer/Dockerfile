# syntax=docker/dockerfile:1
# enable here documents.
# RUN cat <<EOF >> /home/vscode/.bashrc
#
# EOF

FROM ubuntu:latest AS base
# suppress clear apt cache
RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

# Install packages
ENV DEBIAN_FRONTEND=noninteractive

# Set locale and timezone
ENV LANG=ja_JP.UTF-8
ENV LC_ALL=ja_JP.UTF-8
ENV LC_CTYPE=ja_JP.UTF-8
ENV TZ=Asia/Tokyo

ARG USERNAME=vscode
ARG UID=10000
ARG GID=$UID

# Setup user and group id
# Remove all general users and create a new user for USER direction.
RUN set -eux; \
    for i in $(awk -F: '$3 >= 1000 && $3 <= 60000 { print $1 }' /etc/passwd); do \
    userdel $i; \
    done; \
    for i in $(awk -F: '$3 >= 1000 && $3 <= 60000 { print $1 }' /etc/group); do \
    groupdel $i; \
    done; \
    rm -rf /home/*; \
    groupadd --gid "${GID}" "${USERNAME}"; \
    useradd --uid "${UID}" --gid "${GID}" -s /bin/bash -m "${USERNAME}"

# Enable sudo
# RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
#     --mount=type=cache,target=/var/lib/apt,sharing=locked \
#     apt-get update \
#     && apt-get install -y sudo \
#     && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
#     && chmod 0440 /etc/sudoers.d/$USERNAME

# https://github.com/moby/buildkit/issues/1662#issuecomment-683962222
# https://docs.docker.com/reference/dockerfile/?__readwiseLocation=#example-cache-apt-packages
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && apt-get install --no-install-recommends -y \
    # Basic tools
    # ssh is required to handle GitHub in the container
    git ssh \
    # install python
    python-is-python3 \
    # C compiler and assembler for M6502
    cc65 make \
    # System tools
    locales tzdata \
    # Configure locale
    && locale-gen ja_JP.UTF-8 \
    && update-locale LANG=ja_JP.UTF-8 \
    # Configure timezone
    && ln -fs /usr/share/zoneinfo/Asia/Tokyo /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata

FROM base AS dev

ENV DEVCONTAINER_MODE=dev

# install npm
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && apt-get install -y nodejs npm curl ca-certificates \
    python3-venv \
    && npm install n -g \
    && n stable \
    && apt purge -y nodejs npm \
    && apt autoremove -y

# install Codex
ARG CODEX_BUST=0
RUN echo "CODEX_BUST=$CODEX_BUST" \
    && npm i -g @openai/codex

# Default user
USER vscode

FROM base AS runtime

ENV DEVCONTAINER_MODE=runtime

# Default user
USER vscode
