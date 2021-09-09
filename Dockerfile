FROM carlduke/eidos-base:latest

# Install deps
COPY requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt

# Copy main.py
COPY config.py /config.py
COPY dataset/pannuke.py /dataset/pannuke.py
COPY dataset/unitopatho.py /dataset/unitopatho.py
COPY dataset/unitopatho_mask.py /dataset/unitopatho_mask.py
COPY discriminator_model.py /discriminator_model.py
COPY generator_model.py /generator_model.py
COPY utils.py /utils.py
COPY train_utils.py /train_utils.py

#COPY train_utp.py /train.py
COPY train_utp_ddp.py /train.py
COPY test.py /test.py

CMD ["python3", "-u", "/train.py"]
#CMD ["python3", "-u", "/test.py"]