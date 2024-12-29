FROM python:3.12-slim-bullseye

WORKDIR /hypergrad

COPY . .

RUN apt update -y && apt install -y git build-essential cmake 

RUN pip install -r requirements.txt

RUN ./build.sh && ./build/hypergrad_test
    
CMD ["cp", "-r", "./build/hyperpy.cpython-312-aarch64-linux-gnu.so", "/app/build/hyperpy.so"]