# We use the minizinc image as a base
FROM minizinc/minizinc:latest

# Setting the working directory
WORKDIR /src

# Coping all the content of this folder into the container
COPY . .

# Installing python
RUN apt-get update \
  && apt-get install -y python3 \
  && apt-get install -y python3-pip

# Install required libraries
RUN pip install -r requirements.txt
#  && python3 -m pip install -r requirements.txt \

# What to run when the container starts
# Use this command to keep the container up and use the terminal inside of it

# minizinc --solver Gecode nqueens.mzn --json-stream --output-time > results/minizinc/20.json \ && python3 nqueens.py