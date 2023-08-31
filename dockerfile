# Set python version
FROM python:3.10

# Installing CMAKE, g++ and some important libraries

RUN apt-get update && apt-get install -y \
    cmake \
    g++ \
    libmpfr-dev

RUN apt-get update && apt-get install -y libgl1
RUN apt-get update && apt-get install -y libegl1-mesa libegl1

# Downloading MiniZinc

WORKDIR /downloads
ADD https://github.com/MiniZinc/MiniZincIDE/releases/download/2.7.6/MiniZincIDE-2.7.6-bundle-linux-x86_64.tgz MiniZincIDE-2.7.6-bundle-linux-x86_64.tgz
RUN tar -xf MiniZincIDE-2.7.6-bundle-linux-x86_64.tgz
# RUN rm MiniZincIDE-2.2.3-bundle-linux-x86_64.tgz

# Update PATH and LD_LIBRARY_PATH
ENV MZN_DIR="/downloads/MiniZincIDE-2.7.6-bundle-linux-x86_64/"

ENV PATH="${PATH}:${MZN_DIR}/bin"
ENV PATH="${PATH}:${MZN_DIR}/share"
ENV PATH="${PATH}:${MZN_DIR}/lib"


ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${MZN_DIR}/lib"
ENV QT_PLUGIN_PATH="${QT_PLUGIN_PATH}:${MZN_DIR}/plugins"

# Download Geocode

ADD https://github.com/Gecode/gecode/archive/release-6.1.0.tar.gz gecode-6.1.0-source.tar.gz
RUN tar -xf gecode-6.1.0-source.tar.gz

# build gecode
ARG NUM_BUILD_JOBS=1
WORKDIR /downloads/gecode-release-6.1.0
RUN ./configure --disable-examples --disable-gist
RUN make -j ${NUM_BUILD_JOBS}
RUN make install

# Update shared library cache
RUN ldconfig


## Set working directory

WORKDIR /code


# Copy the requirements file

COPY requirements.txt .


# Install dependencies

RUN pip install -r requirements.txt


#Copy the content of the directory

COPY . .


# Run the application

CMD ["python", "./interface.py"]
