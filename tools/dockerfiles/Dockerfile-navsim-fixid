ARG from
FROM ${from}

ENV DEBIAN_FRONTEND=noninteractive
#ENV TZ=UTC
#RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
SHELL ["/bin/bash", "-c"]

ARG duid=1000
ARG dgid=1000
ARG uname=ezdev
ARG gname=ezdev

USER root
RUN id ${uname}
RUN usermod -u ${duid} ${uname} && groupmod -g ${dgid} ${gname}
RUN id ${uname}
#sudo usermod -a -G $DGID ezdev
USER ${uname}
