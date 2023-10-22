FROM python:3.10-slim

RUN apt-get update && apt-get install -y git
RUN pip install jupyter
RUN git clone https://github.com/turi-mate/deepbirding.git

WORKDIR /deepbirding

RUN pip install -r requirements

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]