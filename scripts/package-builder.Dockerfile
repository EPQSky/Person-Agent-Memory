FROM fedora:44

RUN dnf install --assumeyes --setopt=install_weak_deps=False \
      dpkg python3 rpm-build \
    && dnf clean all \
    && rm -rf /var/cache/dnf

WORKDIR /src
ENTRYPOINT ["/bin/bash"]
