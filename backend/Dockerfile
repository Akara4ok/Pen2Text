FROM tensorflow/tensorflow:2.10.0-gpu

WORKDIR /app

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY ./requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY ./Models ./Models
COPY ./Data/SpellCorrection/big.txt ./Data/SpellCorrection/big.txt
COPY ./Data/SpellCorrection/big_ukrainian.txt ./Data/SpellCorrection/big_ukrainian.txt
COPY . .

ENTRYPOINT [ "python3" ]
CMD [ "main.py" ]