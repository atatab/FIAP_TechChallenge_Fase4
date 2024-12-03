import streamlit as st
from datetime import datetime, timedelta
from PIL import Image


# Configuração da página principal
st.set_page_config(
    page_title="FIAP Tech Challenge - Fase 4 - Grupo 52",
    layout="wide"
)

# Barra lateral
with st.sidebar:
    st.markdown("<h2 style='font-size: 30px;'>Análise e Previsão de Preços do Petróleo</h2>", unsafe_allow_html=True)

# Menu de navegação
menu = st.sidebar.selectbox(
    "Navegação",
    [
        "Visão Estratégica do Projeto",
        "Exploração de Dados",
        "Modelo de Previsão",
        "Resumo Executivo do Projeto"
    ]
)

# Páginas
if menu == "Visão Estratégica do Projeto":
    st.title("Visão Estratégica do Projeto")
    st.write("""
    **Objetivo do Projeto**: Fornecer uma análise detalhada e previsões confiáveis sobre os preços do petróleo Brent.
    - **Proposta**: Desenvolvimento de um dashboard interativo e um modelo preditivo para séries temporais.
    - **Ferramentas utilizadas**: Power BI, Visual Studio Code, Python, Streamlit, Pandas, Scikit-learn, entre outras.
    - **Etapas do projeto**:
        1. Coleta e análise exploratória dos dados.
        2. Criação do dashboard com insights estratégicos.
        3. Desenvolvimento de um modelo preditivo.
            """)

    st.write("### Equipe do Projeto")
    st.write("""
    - **Gabriel Moreira do Nascimento**  
    - **Jaqueline Melo da Silva**  
    - **Leandro Cardozo Alves**  
    """)


    # URL da imagem
    image_url = "https://agenciadenoticias.bndes.gov.br/export/sites/default/.galleries/imagegallery/Perspectivas-setor-de-refino-de-petroleo-no-Brasil.jpg"

    # HTML para ajustar largura e altura
    html_code = f"""
    <img src="{image_url}" width="800" height="300" alt="Imagem de Petróleo">
    """

    # Exibir a imagem
    st.markdown(html_code, unsafe_allow_html=True)

elif menu == "Exploração de Dados":
    st.title("Exploração de Dados")

    # Breve descrição sobre o Petróleo Brent
    st.subheader("O que é o Petróleo Brent?")
    st.write("""
    O **Petróleo Brent** é uma referência global para os preços do petróleo bruto, amplamente utilizado como benchmark nos mercados financeiros. 
    Ele é extraído principalmente no Mar do Norte e é conhecido por sua alta qualidade, sendo um óleo leve com baixo teor de enxofre.
    
    O preço do Brent é influenciado por fatores geopolíticos, econômicos e pela oferta e demanda globais de energia, 
    sendo crucial para a tomada de decisão em setores como transporte, energia e finanças.
    """)

    

    # URL da imagem
    image_url = "https://emc.ufsc.br/portal/wp-content/uploads/2014/12/petroleo-e-gas-768x358.jpg"

    # HTML para ajustar largura e altura
    html_code = f"""
    <img src="{image_url}" width="600" height="400" alt="Imagem de Petróleo">
    """

    # Exibir a imagem
    st.markdown(html_code, unsafe_allow_html=True)

    st.write("")

    st.write("### Análise Exploratória e Visualização")
    st.write("""
    Insights identificados:
    1. Correlação do preço com eventos geopolíticos.
    2. Impacto de crises econômicas no preço do petróleo.
    3. Tendências sazonais e de longo prazo.
    4. Mudanças devido à demanda global de energia.
    """)

    st.write("### Contexto")
    st.write("""
    O mercado global de petróleo é marcado por constantes oscilações, onde cada alteração no preço do barril pode gerar impactos profundos na economia mundial. Entender os fatores que influenciam essas variações é essencial para empresas, investidores e governos, permitindo-lhes antecipar tendências, minimizar riscos e aproveitar oportunidades estratégicas.
    Neste trabalho trouxemos a análise de 5 acontecimentos que interferiram no valor do petróleo.
    """)

    # Caminho para a imagem local
    image_path = "assets/geral.png"

    # Abrir a imagem com PIL
    img = Image.open(image_path) 

    # Redimensionar a imagem (ajustando a largura e altura)
    img_resized = img.resize((3000, 1500))  # Ajuste de largura e altura

    # Exibir a imagem redimensionada
    st.image(img_resized, caption="Imagem retirada da Análise no Power BI.", use_container_width=True)

    st.write("**COVID-19**")
    st.write("""
    A pandemia de coronavírus, iniciada em 2020, trouxe impactos significativos a diversos mercados, incluindo o de commodities energéticas. Com o fechamento de fronteiras, a paralisação de fábricas e as medidas de isolamento social, a expectativa de uma drástica redução na demanda levou o preço do Brent a cair 54,25% em março daquele ano. 
    Contudo, a recuperação foi impressionante e ocorreu em pouco tempo. Entre maio e agosto, o valor do Brent saltou 103,75%, com destaque para maio, quando o preço do barril registrou um aumento expressivo de 84,97%. Diversos fatores explicam essa reviravolta.
    A Opep e seus aliados, conhecidos como Opep+, adotaram medidas emergenciais, implementando cortes históricos na produção para estabilizar os preços. Em abril, foi anunciado o maior corte de produção já registrado: 9,7 milhões de barris por dia, o equivalente a 10% da oferta global de petróleo na época.
    Além disso, países fora do cartel também optaram por reduzir voluntariamente sua produção. Paralelamente, muitos governos aproveitaram o período de baixa demanda para ampliar suas reservas estratégicas, absorvendo parte da oferta excedente e contribuindo para a recuperação dos preços.
    """)

    # Caminho para a imagem local
    image_path = "assets/covid.png"

    # Abrir a imagem com PIL
    img = Image.open(image_path) 

    # Redimensionar a imagem (ajustando a largura e altura)
    img_resized = img.resize((3000, 1500))  # Ajuste de largura e altura

    # Exibir a imagem redimensionada
    st.image(img_resized, caption="Imagem retirada da Análise no Power BI.", use_container_width=True)

    st.write("**CRISE ECONOMICA DE 2008**")
    st.write("""
    A crise financeira de 2008, considerada uma das mais graves desde a Grande Depressão de 1929, teve profundos efeitos na economia global. Originada nos Estados Unidos, a crise foi desencadeada pelo colapso do mercado imobiliário, que, nos anos anteriores, havia sido impulsionado por empréstimos hipotecários de alto risco e práticas irresponsáveis de crédito.
    O impacto no mercado de petróleo ocorreu em duas etapas. A primeira, antes da falência do Lehman Brothers em setembro de 2008, viu os preços do barril subirem consideravelmente. Entre abril e junho, o aumento foi de 46%, alcançando o recorde histórico de US$ 147 (R$ 793) por barril. A quebra do banco se tornou um marco simbólico desse período de instabilidade.
    No entanto, a desaceleração econômica global levou a uma queda acentuada nos preços do petróleo, com sete meses seguidos de declínio até o início de 2009. A recuperação começou com a adoção de estímulos financeiros e monetários em larga escala, acompanhados por cortes de produção promovidos pela Opep. Essa combinação deu início à segunda onda de valorização do petróleo: entre fevereiro e junho de 2009, os preços subiram 56,66%, marcando o início de uma retomada gradual no mercado energético.    """)

    # Caminho para a imagem local
    image_path = "assets/crise2008.png"

    # Abrir a imagem com PIL
    img = Image.open(image_path) 

    # Redimensionar a imagem (ajustando a largura e altura)
    img_resized = img.resize((3000, 1500))  # Ajuste de largura e altura

    # Exibir a imagem redimensionada
    st.image(img_resized, caption="Imagem retirada da Análise no Power BI.", use_container_width=True)

    st.write("**GUERRRA DO GOLFO**")
    st.write("""
    A Guerra do Golfo, ocorrida no início da década de 1990, marcou um dos primeiros grandes choques no preço do petróleo desde que os registros do Brent se tornaram mais confiáveis. A invasão do Kuwait pelo Iraque, liderado por Saddam Hussein, gerou temores globais de uma interrupção no fornecimento da commodity.
    Em outubro de 1990, o preço do barril atingiu o pico de US$ 41,90, representando um aumento impressionante de 229% em relação ao fechamento de 1989. Naquele ano, o petróleo encerrou dezembro com uma valorização anual de 30%. A disputa pelo controle da commodity foi o principal motivo da guerra. Saddam Hussein acusava o Kuwait de ultrapassar seus limites de produção e prejudicar os preços.
    Quando o Kuwait recebeu apoio militar dos Estados Unidos em 1991, as forças iraquianas responderam incendiando cerca de 600 campos de petróleo no país vizinho, agravando ainda mais a crise energética e causando danos ambientais devastadores.    """)

    # Caminho para a imagem local
    image_path = "assets/guerradogolfo.png"

    # Abrir a imagem com PIL
    img = Image.open(image_path) 

    # Redimensionar a imagem (ajustando a largura e altura)
    img_resized = img.resize((3000, 1500))  # Ajuste de largura e altura

    # Exibir a imagem redimensionada
    st.image(img_resized, caption="Imagem retirada da Análise no Power BI.", use_container_width=True)

    st.write("**GUERRA DA UCRANIA E RUSSIA**")
    st.write("""
    Uma das crises mais recentes no mercado de petróleo foi desencadeada pela invasão da Ucrânia pela Rússia, que cumpriu sua ameaça após meses de tensões crescentes. Mesmo antes do início do conflito, as incertezas já exerciam pressão sobre o preço do Brent, refletindo o risco iminente de guerra.
    O conflito começou oficialmente em 24 de fevereiro de 2022. Entre dezembro de 2021 e maio de 2022, o preço do barril subiu expressivos 58,13%, com janeiro registrando o maior salto mensal: 18,36% em apenas 30 dias, ainda no período de pré-guerra. Durante o auge da crise, o barril chegou a alcançar US$ 139,13 (R$ 751 no câmbio atual), um dos valores mais altos desde a crise financeira de 2008.
    No total, considerando o período de escalada de tensões em 2021 até a invasão em 2022, o preço do barril acumulou um aumento impressionante de 111,3% ao longo de dois anos, refletindo o impacto profundo das incertezas geopolíticas no mercado global de energia.    """)

    # Caminho para a imagem local
    image_path = "assets/guerraucraniarussia.png"

    # Abrir a imagem com PIL
    img = Image.open(image_path) 

    # Redimensionar a imagem (ajustando a largura e altura)
    img_resized = img.resize((3000, 1500))  # Ajuste de largura e altura

    # Exibir a imagem redimensionada
    st.image(img_resized, caption="Imagem retirada da Análise no Power BI.", use_container_width=True)



    st.write("**GUERRA DE PREÇOS DA OPEP**")
    st.write("""
    Entre janeiro de 1999 e dezembro de 2000, o preço do barril de petróleo registrou um aumento acumulado de 100,3%, com alta de 40% em 1999 e 60% em 2000. Esse período foi marcado por eventos que influenciaram significativamente a oferta e a demanda no mercado global, em meio a uma grave crise financeira nos chamados Tigres Asiáticos Cingapura, Coreia do Sul, Hong Kong e Taiwan, que haviam experimentado rápido crescimento econômico nos anos anteriores.
    Com a desaceleração das economias asiáticas, a Opep reagiu reduzindo a produção em várias ocasiões entre 1998 e 1999, o que sustentou os preços do barril e levou os estoques globais de petróleo a níveis historicamente baixos. Ao mesmo tempo, o cenário global passou por uma recuperação econômica pós-crise, impulsionando a demanda pela commodity e contribuindo para a expressiva valorização no biênio.
    """)

    # Caminho para a imagem local
    image_path = "assets/opep.png"

    # Abrir a imagem com PIL
    img = Image.open(image_path) 

    # Redimensionar a imagem (ajustando a largura e altura)
    img_resized = img.resize((3000, 1500))  # Ajuste de largura e altura

    # Exibir a imagem redimensionada
    st.image(img_resized, caption="Imagem retirada da Análise no Power BI.", use_container_width=True)



elif menu == "Modelo de Previsão":
    st.title("Modelo de Previsão")
    st.write("**Previsão baseada em Machine Learning**")

    # Descrição do modelo
    st.subheader("Sobre o Modelo")
    st.write("""
    Este modelo foi desenvolvido utilizando redes neurais do tipo **LSTM (Long Short-Term Memory)**, 
    uma abordagem avançada para séries temporais. Ele é capaz de capturar padrões complexos nos dados históricos 
    e prever os preços do petróleo Brent com alta precisão.

    ### Como funciona:
    - As redes LSTM são projetadas para aprender dependências de longo prazo, 
      sendo particularmente eficazes para analisar sequências temporais, como preços de mercado.
    - Nosso modelo foi treinado utilizando os dados históricos do petróleo Brent, 
      incluindo fatores como sazonalidade e tendências passadas.

    Insira uma data futura no campo abaixo para visualizar o preço estimado 
    e o comportamento projetado da série temporal.
    """)



    # Valores de exemplo para DATA_INICIAL e LIMITE_DIAS
    DATA_INICIAL = datetime(2023, 1, 1)  # Substitua pela data inicial real
    LIMITE_DIAS = 30  # Substitua pelo número máximo de dias permitido

    # Criando o container para o seletor de data
    with st.container():
        col1, col2 = st.columns([2, 6])  # Criação de duas colunas com larguras diferentes
        with col1:
            min_date = DATA_INICIAL + timedelta(days=1)
            max_date = DATA_INICIAL + timedelta(days=LIMITE_DIAS)
            end_date = st.date_input(
                "Escolha a data de previsão:", 
                min_value=min_date, 
                max_value=max_date,
                value=min_date,
            )
        with col2:
            st.write(f"Data selecionada: {end_date}")

elif menu == "Resumo Executivo do Projeto":
    st.title("Resumo Executivo do Projeto")

   # Introdução
    st.subheader("Resultados do Projeto")
    st.write("""
    Este projeto teve como objetivo principal fornecer insights estratégicos sobre os preços do petróleo Brent 
    e desenvolver uma solução preditiva baseada em Machine Learning. Abaixo, resumimos os principais resultados e entregas alcançados.
    """)

    # Principais Realizações
    st.subheader("Principais Realizações")
    st.write("""
    - **Dashboard Interativo**: Desenvolvido com foco em storytelling, permitindo a análise de tendências históricas e seus fatores influenciadores.
    - **Insights Estratégicos**: Identificamos quatro insights principais sobre variações no preço do petróleo, incluindo:        
        1. Impactos de eventos geopolíticos globais.
        2. Reações a crises econômicas e políticas.
        3. Tendências sazonais e ciclos de mercado.
        4. Mudanças na demanda global por energia.
    - **Modelo de Previsão de Preços**: Criamos um modelo preditivo funcional, capaz de prever os preços do petróleo diariamente com uma margem de erro de [insira a métrica de erro aqui, como RMSE ou MAPE].
    - **Deploy do MVP**: O modelo foi integrado ao Streamlit como uma aplicação interativa para suporte à tomada de decisão.
    """)



# Rodapé ou informações adicionais
st.sidebar.info("Tech Challenge - Consultoria para Previsão de Preços do Petróleo")
