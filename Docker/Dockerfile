FROM ubuntu:22.04

# 1. Basic setup
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y \
    git cmake build-essential qtbase5-dev libqt5x11extras5-dev \
    libglew-dev libboost-all-dev python3-dev python3-pip libpython3-dev \
    libeigen3-dev libassimp-dev libvtk7-dev \
    libgl1-mesa-dev libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 \
    libxcb-keysyms1 libxcb-render-util0 libxcb-xinerama0 libxcb-xinput0 \
    mesa-utils x11-apps \
    && pip3 install numpy pandas matplotlib \
    && apt clean

# 2. Get SOFA
WORKDIR /opt
RUN git clone -b v24.12.00 https://github.com/sofa-framework/sofa.git sofa/src

# 3. Clone plugins
RUN mkdir -p sofa/applications/plugins
WORKDIR /opt/sofa/applications/plugins
RUN git clone https://github.com/SofaDefrost/SoftRobots.git

# 4. Configure with CMake
WORKDIR /opt/sofa
RUN mkdir build
WORKDIR /opt/sofa/build
RUN cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DSOFA_BUILD_METIS=ON \
  -DSOFA_BUILD_SOFTROBOTS=ON \
  -DSOFA_BUILD_STLIB=ON \
  -DSOFA_BUILD_SCENELOADER=ON \
  -DSOFA_BUILD_GUI_QT=ON \
  -DSOFA_BUILD_TESTS=OFF

# 5. Build
RUN make -j$(nproc)

# 6. Set entrypoint
WORKDIR /opt/sofa/build
ENTRYPOINT [ "./bin/runSofa" ]
