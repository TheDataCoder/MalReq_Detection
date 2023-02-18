FROM continuumio/anaconda3
COPY . /usr/app/
EXPOSE 5000
WORKDIR /usr/app/
RUN conda create -n py38 python=3.8.8 -y
RUN echo "source activate py38" > ~/.bashrc
RUN conda install -c conda-forge hdbscan==0.8.28 -y
RUN conda install -c conda-forge scikit-learn==1.1.1 -y
RUN pip install -r requirements.txt
WORKDIR src
CMD python predict_flasgger.py
EXPOSE 5000