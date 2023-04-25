FROM mirrors.tencent.com/taiji_light/g-tlinux2.2-python3.6-cuda10.2-cudnn8.0:latest

ARG CONDA_VER=latest
ARG PY_VER=3.10

# Install taiji_client and ceph-fuse
# Reference:
#   https://iwiki.woa.com/pages/viewpage.action?pageId=777831494
#   https://iwiki.woa.com/pages/viewpage.action?pageId=975928529
RUN wget http://jizhi.oa.com/taiji_client_golang/taiji_client -O /usr/bin/taiji_client && chmod +x /usr/bin/taiji_client
RUN mkdir /etc/ceph/ && touch /etc/ceph/ceph.conf
RUN echo -e \
    $"[client] \n\
    client_not_support_security = true \n\
    client_setuid_optimize = true \n\
    fuse_fake_tmp_agent = true \n\
    client_trash_enabled = false \n\
    client_reconnect_stale = true \n\
    fuse_attr_timeout = 5 \n\
    client_cache_size = 30000 \n\
    client_die_on_failed_dentry_invalidate = false \n\
    fuse_set_user_groups = false \n\
    fuse_clone_fd = false \n\
    objecter_max_osd_sessions = 80" >> /etc/ceph/ceph.conf
RUN curl -o /etc/yum.repos.d/ceph_el7.repo http://gaia.repo.oa.com/ceph_el7.repo && yum install ceph-fuse --enablerepo=ceph-luminous -y
# RUN taiji_client mount -l sz -tk ${TOKEN} /mnt/private_${USER}

# Install htop, unzip, etc.
RUN yum install -y htop unzip && yum clean all && rm -rf /var/cache/yum

# Install miniconda to /miniconda
RUN wget --no-check-certificate "http://repo.continuum.io/miniconda/Miniconda3-${CONDA_VER}-Linux-x86_64.sh"
RUN bash Miniconda3-${CONDA_VER}-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-${CONDA_VER}-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda && conda init

# Install packages from conda
RUN conda install -c anaconda -y python=${PY_VER}
RUN conda install -c pytorch -y \
    pytorch==1.12.1 \
    torchaudio==0.12.1 \
    cudatoolkit=10.2 \
    && conda clean --yes --index-cache --tarballs --tempfiles --logfiles
RUN conda install tensorboard -y && conda clean --yes --index-cache --tarballs --tempfiles --logfiles

# enter work dir
WORKDIR /app

# install python dependences
COPY ./requirements.txt /app
RUN pip install requests tornado
RUN pip install --no-cache-dir --upgrade -r requirements.txt
