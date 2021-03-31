# Each instruction creates a new layer of the image.
FROM ubuntu:16.04

WORKDIR /webserver/telemarket
COPY . .
ENV LANG='C.UTF-8' LC_ALL='C.UTF-8' TZ='Asia/Shanghai'
RUN printf "nameserver 8.8.8.8\nnameserver 8.8.4.4\nnameserver 114.114.114.114\n" > /etc/resolv.conf \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && find . -name '*.pyc' -exec rm {} \; \
    && sed -i s@security.ubuntu.com@mirrors.aliyun.com@g /etc/apt/sources.list \
    && sed -i s@archive.ubuntu.com@mirrors.aliyun.com@g /etc/apt/sources.list \
    && sed -i s@us.archive.ubuntu.com@mirrors.aliyun.com@g /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    iputils-ping \
    python3 \
    python3-pip \
    libffi6 \
    libffi-dev \
    aria2 \
    xz-utils \
    gcc \
    python3-dev \
    python3-setuptools \
    libmysqlclient20 \
    libmysqlclient-dev \
    libjpeg8-dev \
    libjpeg8 \
    libfreetype6-dev \
    libfreetype6 \
    && apt-get -y -o Dpkg::Options::="--force-confmiss" install --reinstall netbase \
    && pip3 install -i http://xxx/repository/pypi-proxy-aliyun/simple/ --no-cache-dir --trusted-host xxx  -r requirements.txt \
    && pip3 install -i http://xxx/repository/pypi-proxy-aliyun/simple/ --no-cache-dir --trusted-host xxx -r requirements1.txt  \
    && yes | cp -rf /webserver/telemarket/others/__init__.py  /usr/local/lib/python3.5/dist-packages/flask_alchemydumps/ \
    && apt-get purge -y \
    libffi-dev \
    aria2 \
    xz-utils \
    gcc \
    python3-dev \
    libmysqlclient-dev \
    libjpeg8-dev \
    libfreetype6-dev \
    && apt-get purge --auto-remove -y \
    && apt-get autoclean -y \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p flasgger/flasgger_static \
    && cp -rf /usr/local/lib/python3.5/dist-packages/flasgger/ui2/static/* ./flasgger/flasgger_static/ \
    && export PYTHONPATH=${PYTHONPATH}:/webserver/telemarket

EXPOSE 8200
VOLUME /webserver/telemarket
CMD ["gunicorn", "-w", "4", "-b", ":8200", "app:app"]