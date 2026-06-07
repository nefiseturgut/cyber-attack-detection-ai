FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    apache2 \
    openssh-server \
    net-tools \
    iputils-ping \
    iproute2 \
    && apt-get clean

RUN mkdir -p /var/run/sshd
RUN useradd -m -s /bin/bash labuser && \
    echo "labuser:password123" | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    echo "root:toor" | chpasswd

RUN echo '<html><body><h1>IDS Lab - Target</h1></body></html>' > /var/www/html/index.html

COPY start_target.sh /start_target.sh
RUN chmod +x /start_target.sh

EXPOSE 22 80
ENTRYPOINT ["/start_target.sh"]