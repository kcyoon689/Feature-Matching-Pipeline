FROM ubuntu:22.04

# zone
ENV TIME_ZONE="Asia/Seoul"
RUN ln -snf /usr/share/zoneinfo/${TIME_ZONE} /etc/localtime	&& echo ${TIME_ZONE} > /etc/timezone

# user-setting
ARG USER_NAME="yoonk"
ARG USER_UID="1000"
ARG USER_GID=${USER_UID}
RUN groupadd --gid ${USER_GID} ${USER_NAME} \
	&& useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USER_NAME} \
	&& apt-get -qq update \
	&& apt-get -qq install -y sudo \
    && apt-get -qq install -y git \
    && apt-get -qq install -y wget
	# && echo ${USER_NAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USER_NAME} \
	# && echo ${USER_NAME} ALL=\(root\) NOPASSWD:ALL >> /etc/sudoers \
	# && chmod 0440 /etc/sudoers.d/${USER_NAME}

# python
RUN apt-get -qq update \
	&& apt-get -qq install -y python3-dev \
	&& apt-get -qq install -y python3-pip \
	&& python3 -m pip install --upgrade pip \
	&& python3 -m pip install --root-user-action=ignore requests \
    && python3 -m pip install kornia==0.7.0 \
    && python3 -m pip install kornia_moons==0.2.6 \
    && python3 -m pip install opencv-python-headless \
    && python3 -m pip install torch==1.12.1 \
    && python3 -m pip install imutils==0.5.4 \
    && python3 -m pip install scipy==1.11.2 \
    && python3 -m pip install tqdm \
    && python3 -m pip install pandas \
    && python3 -m pip install plotly

# RUN export PIP_ROOT_USER_ACTION=ignore

RUN apt-get -qq update \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/* \
	&& rm -rf /var/tmp/* \
	&& rm -rf /tmp/*

USER ${USER_NAME}

# git
# RUN apt-get -qq install -y git
# RUN git config --global pager.branch false

# shell
# RUN sudo apt-get -qq update \
# 	&& sudo apt-get -qq install -y zsh wget \
# 	&& wget "https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh" -O - | zsh || true \
# 	&& git clone "https://github.com/zsh-users/zsh-autosuggestions" ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions \
# 	&& git clone "https://github.com/zsh-users/zsh-syntax-highlighting.git" ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting \
# 	&& git clone "https://github.com/zsh-users/zsh-completions" ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-completions \
# 	&& git clone "https://github.com/supercrabtree/k" ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/k
# ADD .zshrc /home/${USER_NAME}
# ENV SHELL "/bin/zsh"