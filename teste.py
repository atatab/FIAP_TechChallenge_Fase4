import streamlit as st

# Configuração da página principal
st.set_page_config(
    page_title="Análise e Previsão de Preços do Petróleo",
    layout="wide"
)

# Menu de navegação
menu = st.sidebar.selectbox(
    "Navegação",
    [
        "Visão Estratégica do Projeto",
        "Exploração de Dados e Insights",
        "Modelo de Previsão de Preços",
        "Conclusões e Recomendações Estratégicas"
    ]
)

# Páginas
if menu == "Visão Estratégica do Projeto":
    st.title("Visão Estratégica do Projeto")
    st.write("""
    **Objetivo do Projeto**: Fornecer uma análise detalhada e previsões confiáveis sobre os preços do petróleo Brent.
    - **Proposta**: Desenvolvimento de um dashboard interativo e um modelo preditivo para séries temporais.
    - **Ferramentas utilizadas**: Python, Streamlit, Pandas, Scikit-learn, entre outras.
    - **Etapas do projeto**:
        1. Coleta e análise exploratória dos dados.
        2. Criação do dashboard com insights estratégicos.
        3. Desenvolvimento de um modelo preditivo.
        4. Deploy e apresentação do MVP.
    """)

elif menu == "Exploração de Dados e Insights":
    st.title("Exploração de Dados e Insights")
    st.write("**Análise Exploratória e Visualização**")
    # Inserir gráficos e tabelas aqui
    st.write("""
    Insights identificados:
    1. Correlação do preço com eventos geopolíticos.
    2. Impacto de crises econômicas no preço do petróleo.
    3. Tendências sazonais e de longo prazo.
    4. Mudanças devido à demanda global de energia.
    """)
    # Exemplo de gráfico
    st.line_chart([10, 20, 30, 40, 50])  # Substituir com dados reais

elif menu == "Modelo de Previsão de Preços":
    st.title("Modelo de Previsão de Preços")
    st.write("**Previsão baseada em Machine Learning**")
    st.write("""
    - O modelo desenvolvido utiliza algoritmos como ARIMA, LSTM ou Prophet para previsão de séries temporais.
    - Performance do modelo:
        - RMSE: [valor]
        - MAPE: [valor]
    - Código e detalhes do modelo estão integrados ao dashboard.
    """)
    # Exemplo de visualização de previsões
    st.line_chart([50, 52, 54, 55, 53])  # Substituir com dados preditivos reais

elif menu == "Conclusões e Recomendações Estratégicas":
    st.title("Conclusões e Recomendações Estratégicas")
    st.write("""
    **Principais Conclusões**:
    - O modelo preditivo fornece previsões confiáveis com baixa margem de erro.
    - O dashboard é uma ferramenta poderosa para suporte à tomada de decisão.
    
    **Recomendações**:
    1. Monitorar continuamente as entradas de dados e atualizar o modelo.
    2. Expandir o escopo para incluir variáveis externas, como inflação e preços de commodities relacionadas.
    3. Implementar uma solução robusta de deploy para integração com sistemas do cliente.
    """)

# Rodapé ou informações adicionais
st.sidebar.info("Tech Challenge - Consultoria para Previsão de Preços do Petróleo")
