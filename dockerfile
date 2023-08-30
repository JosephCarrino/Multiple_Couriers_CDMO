















# Set python version

FROM python:3.10



# Set working directory

WORKDIR /code


# Copy the requirements file

COPY requirements.txt .


# Install dependencies

RUN pip install -r requirements.txt


#Copy the content of the directory

COPY . .


# Run the application

CMD ["python", "./interface.py"]
