FROM python:3.10.12

RUN apt-get update && apt-get install -y git
RUN pip install jupyter
RUN git clone https://github.com/turi-mate/deepbirding.git

RUN mkdir -p $HOME/deepbirding
WORKDIR $HOME/deepbirding

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
