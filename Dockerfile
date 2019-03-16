FROM ubuntu:16.04
MAINTAINER Mehdi Manouchehri <Manooch.github.io>

RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install numpy pandas matplotlib plotly scikit-learn

ADD breastcanceranalysis.py /

ENTRYPOINT ["python3"]
CMD ["./breastcanceranalysis.py"]
