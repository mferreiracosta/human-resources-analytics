# Carregando as bibliotecas
from typing_extensions import dataclass_transform
import pandas as pd
import streamlit as st
from minio import Minio 
import joblib 
import matplotlib.pyplot as plt
from pycaret.classification import load_model, predict_model

# Baixando os arquivos do Data Lake
client = Minio(
        endpoint="172.21.160.1:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        secure=False
    )

# Modelo de classificação, dataset e cluster
client.fget_object("curated", "model.pkl", "model.pkl")
client.fget_object("curated", "dataset.csv", "dataset.csv")
client.fget_object("curated", "cluster.joblib", "cluster.joblib")

var_model = "model"
var_model_cluster = "cluster.joblib"
var_dataset = "dataset.csv"

# Carregando o modelo treinado
model = load_model(var_model)
model_cluster = joblib.load(var_model_cluster)

# Carregando o conjunto de dados
dataset = pd.read_csv(var_dataset)

print(dataset.head())

# Códigos Streamlit - DataApp
# Título
st.title("Human Resources Analytics")

# Subtítulo
st.markdown("Este é um DataApp utilizado para exibir a solução de Machine Learning para o problema de Human Resources")

# Imprime o conjunto de dados
st.dataframe(dataset.head())

# Grupos de empregados
kmeans_colors = ['green' if c == 0 else 'red' if c == 1 else 'blue' for c in model_cluster.labels_]

st.sidebar.subheader("Defina os atributos do empregado para predição de turnover")

# Mapeando dados do usuário para cada atributo
satisfaction = st.sidebar.number_input("satisfaction", value=dataset["satisfaction"].mean())
evaluation = st.sidebar.number_input("evaluation", value=dataset["evaluation"].mean())
averageMonthlyHours = st.sidebar.number_input("averageMonthlyHours", value=dataset["averageMonthlyHours"].mean())
yearsAtCompany = st.sidebar.number_input("yearsAtCompany", value=dataset["yearsAtCompany"].mean())

# Inserindo um botão na tela
btn_predict = st.sidebar.button("Realizar Classificação")

# Verifica se o botão foi acionado
if btn_predict:
    data_teste = pd.DataFrame()
    data_teste["satisfaction"] = [satisfaction]
    data_teste["evaluation"] = [evaluation]
    data_teste["averageMonthlyHours"] = [averageMonthlyHours]
    data_teste["yearsAtCompany"] = [yearsAtCompany]

    # Imprime os dados de teste
    print(data_teste)

    # Realiza a predição
    result = predict_model(model, data=data_teste)

    st.write(result)

    fig = plt.figure(figsize=(10,6))
    plt.scatter(x="satisfaction",
                y="evaluation",
                data=dataset[dataset.turnover==1],
                alpha=0.25,
                color=kmeans_colors)

    plt.xlabel("Satisfaction")
    plt.ylabel("Evaluation")

    plt.scatter(x=model_cluster.cluster_centers_[:,0],
                y=model_cluster.cluster_centers_[:,1],
                color="black",
                marker="X",
                s=100)

    plt.scatter(x=[satisfaction],
                y=[evaluation],
                color="yellow",
                marker="X",
                s=300)
    
    plt.title("Grupos de Empregados - Satisfação vs Avaliação")
    plt.show()
    st.pyplot(fig)