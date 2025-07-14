FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# user-setting
ARG USER_NAME="user"
ARG USER_UID="1000"
ARG USER_GID=${USER_UID}

# zone
ENV TIME_ZONE="Asia/Seoul"
RUN ln -snf /usr/share/zoneinfo/${TIME_ZONE} /etc/localtime	&& echo ${TIME_ZONE} > /etc/timezone

RUN apt-get update \
	&& apt-get install -y --no-install-recommends sudo apt-utils x11-apps \
 	&& apt-get install -y build-essential cmake \
	&& apt-get install -y zsh net-tools wget curl git \
	&& apt-get install -y v4l-utils ffmpeg usbutils \
	&& apt-get install -y python3-dev python3-pip python3-venv \
 	# && apt-get install -y alsa-utils portaudio19-dev python3-tk pulseaudio \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/*

RUN echo "USER_NAME=${USER_NAME}, UID=${USER_UID}, GID=${USER_GID}"


RUN groupadd --gid ${USER_GID} ${USER_NAME} || true \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USER_NAME} \
    && apt-get update \
    && echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USER_NAME} \
    && chmod 0440 /etc/sudoers.d/${USER_NAME}




# python
RUN apt-get -qq update \
	&& apt-get -qq install -y python3-dev \
	&& apt-get -qq install -y python3-pip \
	&& python3 -m pip install --upgrade pip \
	&& python3 -m pip install --root-user-action=ignore requests \
    && python3 -m pip install kornia==0.7.0 \
    && python3 -m pip install kornia_moons==0.2.6 \
    && python3 -m pip install opencv-python\
	&& python3 -m pip install opencv-contrib-python\
    && python3 -m pip install torch==1.12.1 \
	&& python3 -m pip install torchvision==0.13.1 \
    && python3 -m pip install imutils==0.5.4 \
    && python3 -m pip install scipy==1.11.2 \
    && python3 -m pip install tqdm \
    && python3 -m pip install pandas \
    && python3 -m pip install plotly \
	&& python3 -m pip install tenacity==8.3.0 \
	&& python3 -m pip install pillow==9.5.0 \
	&& python3 -m pip install pyyaml \
    && python3 -m pip install omegaconf

# RUN export PIP_ROOT_USER_ACTION=ignore

RUN apt-get -qq update \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/* \
	&& rm -rf /var/tmp/* \
	&& rm -rf /tmp/*

USER ${USER_NAME}

# shell
# Zsh 및 oh-my-zsh 설치
RUN sudo apt-get -qq update \
	&& sudo apt-get -qq install -y zsh wget \
	&& wget "https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh" -O - | zsh || true \
	&& git clone "https://github.com/zsh-users/zsh-autosuggestions" ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions \
	&& git clone "https://github.com/zsh-users/zsh-syntax-highlighting.git" ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting \
	&& git clone "https://github.com/zsh-users/zsh-completions" ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-completions \
	&& git clone "https://github.com/supercrabtree/k" ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/k

ADD .zshrc /home/${USER_NAME}

# 기본 쉘을 Zsh로 설정
ENV SHELL "/bin/zsh"