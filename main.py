import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


st.set_page_config(page_title="Dashboard Comercial ", layout="wide", page_icon="🛒")
st.markdown("""
<style>
.metric-card { background-color: #ffffff10; padding: 12px; border-radius: 10px; text-align:center; border:1px solid #00000010; }
.section-title { font-size:20px; font-weight:600; margin-bottom:6px; }
.small-muted { color:#8c8c8c; font-size:12px; }
</style>
""", unsafe_allow_html=True)


DATA_PATH = "dados_vendas.csv"   # caminho do arquivo que você enviou

@st.cache_data
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    # garantir tipos e colunas
    df['Data'] = pd.to_datetime(df['Data'])
    # corrigir ValorTotal pequeno desvio
    if {'Quantidade', 'PrecoPorUnidade', 'ValorTotal'}.issubset(df.columns):
        calc = df['Quantidade'] * df['PrecoPorUnidade']
        diff = (df['ValorTotal'] - calc).abs() > 0.01
        if diff.any():
            df.loc[diff, 'ValorTotal'] = calc.loc[diff]
    # colunas auxiliares
    df['AnoMes'] = df['Data'].dt.to_period('M').astype(str)
    # dia da semana 
    df['DiaSemana'] = df['Data'].dt.day_name() 
    dia_map = {
        'Monday': 'Segunda', 'Tuesday': 'Terça', 'Wednesday': 'Quarta',
        'Thursday': 'Quinta', 'Friday': 'Sexta', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
    }
    df['DiaSemana_pt'] = df['DiaSemana'].map(dia_map)
    return df

df = load_data()

st.sidebar.header("Filtros")

# período
min_date = df['Data'].min().date()
max_date = df['Data'].max().date()
date_range = st.sidebar.date_input("Período", [min_date, max_date], help="Selecione o intervalo de datas")


meses = sorted(df['Mes'].unique())
opcoes_meses = ["Todos"] + [str(m) for m in meses]

escolha_meses = st.sidebar.multiselect(
    "Mês (número)",
    options=opcoes_meses,
    default="Todos"
)

if "Todos" in escolha_meses:
    filtro_meses = meses
else:
    filtro_meses = [int(m) for m in escolha_meses]



# categorias
categorias = sorted(df['CategoriaProduto'].unique())
filtro_categorias = st.sidebar.multiselect("Categoria", options=categorias, default=categorias)

# gênero
generos = sorted(df['Genero'].unique())
filtro_generos = st.sidebar.multiselect("Gênero", options=generos, default=generos)

# faixa etária
faixas = sorted(df['FaixaEtaria'].unique())
filtro_faixas = st.sidebar.multiselect("Faixa Etária", options=faixas, default=faixas)


mask = (
    (df['Data'].dt.date >= date_range[0]) &
    (df['Data'].dt.date <= date_range[1]) &
    (df['Mes'].isin(filtro_meses)) &
    (df['CategoriaProduto'].isin(filtro_categorias)) &
    (df['Genero'].isin(filtro_generos)) &
    (df['FaixaEtaria'].isin(filtro_faixas))
)
df_f = df[mask].copy()

if df_f.empty:
    st.warning("Nenhum registro encontrado com os filtros aplicados. Ajuste os filtros na barra lateral.")

st.title("Dashboard Comercial usando Dataset Dados_Vendas.csv")

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Receita Total ", f"R$ {df_f['ValorTotal'].sum():,.2f}")
    st.markdown('</div>', unsafe_allow_html=True)
with k2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Ticket Médio", f"R$ {df_f['ValorTotal'].mean():,.2f}")
    st.markdown('</div>', unsafe_allow_html=True)
with k3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Transações", f"{len(df_f)}")
    st.markdown('</div>', unsafe_allow_html=True)
with k4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Clientes únicos", f"{df_f['IDCliente'].nunique()}")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

tab_visao, tab_perfil, tab_cat, tab_temp, tab_corr = st.tabs([
    "Visão Geral", "Perfil do Cliente", "Categorias", "Temporal", "Correlação"
])

with tab_visao:
    st.header("Visão Geral")
    st.write("Resumo rápido com os insights mais importantes e gráficos interativos.")

    if not df_f.empty and not df_f.empty:
        cat_rev = df_f.groupby('CategoriaProduto')['ValorTotal'].sum().sort_values(ascending=False)
        top_cat = cat_rev.index[0]
        top_cat_val = cat_rev.iloc[0]

        # cliente top
        cliente_rev = df_f.groupby('IDCliente')['ValorTotal'].sum().sort_values(ascending=False)
        top_cliente = cliente_rev.index[0]
        top_cliente_val = cliente_rev.iloc[0]

        # melhor dia (receita)
        dia_rev = df_f.groupby('DiaSemana_pt')['ValorTotal'].sum().sort_values(ascending=False)
        top_dia = dia_rev.index[0]

        # gênero com maior gasto médio
        gen_avg = df_f.groupby('Genero')['ValorTotal'].mean().sort_values(ascending=False)
        top_genero = gen_avg.index[0]
        top_genero_val = gen_avg.iloc[0]

        # faixa etária mais valiosa (soma de ValorTotal)
        faixa_rev = df_f.groupby('FaixaEtaria')['ValorTotal'].sum().sort_values(ascending=False)
        top_faixa = faixa_rev.index[0]

        # crescimento por categoria (simples: diferença % entre 1º e último mês do período)
        rev_cat_mes = df_f.groupby(['CategoriaProduto','AnoMes'])['ValorTotal'].sum().reset_index()
        # escolher primeiro e ultimo mes no dataset filtrado (global)
        meses_ordenados = sorted(df_f['AnoMes'].unique())
        if len(meses_ordenados) >= 2:
            primeiro = meses_ordenados[0]
            ultimo = meses_ordenados[-1]
            start = rev_cat_mes[rev_cat_mes['AnoMes']==primeiro].set_index('CategoriaProduto')['ValorTotal']
            end = rev_cat_mes[rev_cat_mes['AnoMes']==ultimo].set_index('CategoriaProduto')['ValorTotal']
            growth = ((end - start) / start.replace({0: np.nan})).dropna().sort_values(ascending=False)
            top_growth_cat = growth.index[0] if not growth.empty else None
        else:
            top_growth_cat = None
    else:
        top_cat = top_cliente = top_dia = top_genero = top_faixa = top_growth_cat = None
        top_cat_val = top_cliente_val = top_genero_val = 0.0

    c1, c2, c3 = st.columns(3)
    c1.info(f"Categoria mais lucrativa: **{top_cat}** (R$ {top_cat_val:,.2f})")
    c2.info(f"Cliente que mais gastou: **{top_cliente}** (R$ {top_cliente_val:,.2f})")
    c3.info(f"Melhor dia (receita): **{top_dia}**")

    c4, c5, c6 = st.columns(3)
    c4.info(f"Gênero com maior gasto médio: **{top_genero}** (ticket médio R$ {top_genero_val:,.2f})")
    c5.info(f"Faixa etária mais valiosa: **{top_faixa}**")
    c6.info(f"Categoria com maior crescimento: **{top_growth_cat}**")

    st.markdown("----")
    # Gráfico: receita por AnoMes
    st.subheader("Receita por Ano-Mês")
    receita_am = df_f.groupby('AnoMes')['ValorTotal'].sum().reset_index()
    fig_am = px.line(receita_am, x='AnoMes', y='ValorTotal', markers=True)
    fig_am.update_layout(xaxis_title="Ano-Mês", yaxis_title="Receita (R$)")
    st.plotly_chart(fig_am, use_container_width=True)

    # Treemap receita por categoria (fácil visual)
    st.subheader("Receita por Categoria (treemap)")
    rev_cat = df_f.groupby('CategoriaProduto')['ValorTotal'].sum().reset_index().sort_values('ValorTotal', ascending=False)
    fig_tree = px.treemap(rev_cat, path=['CategoriaProduto'], values='ValorTotal')
    st.plotly_chart(fig_tree, use_container_width=True)

# -------------------------
# Tab 2: Perfil do Cliente
# -------------------------
with tab_perfil:
    st.header("Perfil do Cliente")
    st.write("Comportamento dos clientes, frequência e top clientes — útil para segmentação")

    # Top 10 clientes por receita
    st.subheader("Top 10 Clientes por Receita")
    top_clients = df_f.groupby('IDCliente')['ValorTotal'].sum().reset_index().sort_values('ValorTotal', ascending=False).head(10)
    fig_topc = px.bar(top_clients, x='IDCliente', y='ValorTotal', title='Top 10 Clientes por Receita')
    st.plotly_chart(fig_topc, use_container_width=True)

    # Frequência: transações por cliente (histograma)
    st.subheader("Frequência de compra por cliente")
    freq = df.groupby('IDCliente')['IDTransacao'].count().reset_index().rename(columns={'IDTransacao':'Frequencia'})
    fig_freq = px.histogram(freq, x='Frequencia', nbins=20, title='Distribuição de frequência de compras por cliente')
    st.plotly_chart(fig_freq, use_container_width=True)

    # Distribuição por faixa etária (contagem de clientes únicos)
    st.subheader("Distribuição de clientes por Faixa Etária")
    clientes_faixa = df.groupby('FaixaEtaria')['IDCliente'].nunique().reset_index().rename(columns={'IDCliente':'ClientesUnicos'})
    fig_cli_faixa = px.bar(clientes_faixa, x='FaixaEtaria', y='ClientesUnicos', title='Clientes únicos por Faixa Etária')
    st.plotly_chart(fig_cli_faixa, use_container_width=True)

    # Ticket médio por gênero e faixa
    st.subheader("Ticket médio por Gênero")
    ticket_gen = df.groupby('Genero')['ValorTotal'].mean().reset_index().sort_values('ValorTotal', ascending=False)
    fig_tgen = px.bar(ticket_gen, x='Genero', y='ValorTotal')
    st.plotly_chart(fig_tgen, use_container_width=True)


with tab_cat:
    st.header("Análise por Categoria")
    st.write("Foco em categorias: volume, receita e comportamento.")

    # Quantidade total por categoria
    st.subheader("Quantidade total por Categoria")
    qtd_cat = df.groupby('CategoriaProduto')['Quantidade'].sum().reset_index().sort_values('Quantidade', ascending=False)
    fig_qcat = px.bar(qtd_cat, x='CategoriaProduto', y='Quantidade')
    st.plotly_chart(fig_qcat, use_container_width=True)

    # Ticket médio por categoria
    st.subheader("Ticket médio por Categoria")
    ticket_cat = df.groupby('CategoriaProduto')['ValorTotal'].mean().reset_index().sort_values('ValorTotal', ascending=False)
    fig_tcat = px.bar(ticket_cat, x='CategoriaProduto', y='ValorTotal')
    st.plotly_chart(fig_tcat, use_container_width=True)

    # Receita por categoria ao longo do tempo (linha empilhada)
    st.subheader("Receita por Categoria ao longo do tempo")
    rev_cat_time = df.groupby(['AnoMes','CategoriaProduto'])['ValorTotal'].sum().reset_index()
    fig_stack = px.area(rev_cat_time, x='AnoMes', y='ValorTotal', color='CategoriaProduto')
    st.plotly_chart(fig_stack, use_container_width=True)


with tab_temp:
    st.header("Análise Temporal")
    st.write("Comportamento por mês, dia da semana e fim de semana.")

    # Receita por mês (soma)
    st.subheader("Receita por Mês (numérico)")
    receita_mes = df.groupby('Mes')['ValorTotal'].sum().reset_index()
    fig_mes = px.bar(receita_mes, x='Mes', y='ValorTotal')
    st.plotly_chart(fig_mes, use_container_width=True)

    # Receita por dia da semana
    st.subheader("Receita por Dia da Semana")
    dia_rev = df.groupby('DiaSemana_pt')['ValorTotal'].sum().reset_index()
    # order weekdays
    order = ['Segunda','Terça','Quarta','Quinta','Sexta','Sábado','Domingo']
    dia_rev['DiaSemana_pt'] = pd.Categorical(dia_rev['DiaSemana_pt'], categories=order, ordered=True)
    dia_rev = dia_rev.sort_values('DiaSemana_pt')
    fig_dia = px.bar(dia_rev, x='DiaSemana_pt', y='ValorTotal')
    st.plotly_chart(fig_dia, use_container_width=True)

    # Fim de semana vs dias úteis (ticket médio)
    st.subheader("Ticket médio: Fim de semana vs Dia útil")
    avg_weekend = df[df['EhFimDeSemana']==1]['ValorTotal'].mean()
    avg_weekday = df[df['EhFimDeSemana']==0]['ValorTotal'].mean()
    df_w = pd.DataFrame({
        'Tipo':['Dia útil','Fim de semana'],
        'Ticket':[avg_weekday if not np.isnan(avg_weekday) else 0, avg_weekend if not np.isnan(avg_weekend) else 0]
    })
    fig_week = px.bar(df_w, x='Tipo', y='Ticket')
    st.plotly_chart(fig_week, use_container_width=True)


with tab_corr:
    st.header("Comparações e Correlações")
    st.write("Relações fáceis de interpretar entre as principais variáveis.")

    # Correlação numérica
    num_cols = ['Idade','Quantidade','PrecoPorUnidade','ValorTotal']
    available = [c for c in num_cols if c in df.columns]
    if len(available) >= 2:
        corr = df[available].corr()
        fig_corr = px.imshow(corr, text_auto=True, title='Mapa de Correlação (numéricas)')
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Não há colunas numéricas suficientes para correlação.")

    # Scatter interativo: Quantidade x ValorTotal (com cor por categoria)
    st.subheader("Quantidade x ValorTotal (por Categoria)")
    fig_sc = px.scatter(df, x='Quantidade', y='ValorTotal', color='CategoriaProduto', hover_data=['IDCliente','FaixaEtaria'], title='Quantidade vs ValorTotal')
    st.plotly_chart(fig_sc, use_container_width=True)

    # Idade vs Ticket médio 
    if 'Idade' in df.columns:
        st.subheader("Idade vs Ticket médio")
        df['Idade_binned'] = pd.cut(df['Idade'], bins=[0,18,25,35,45,60,100], labels=['0-18','19-25','26-35','36-45','46-60','60+'])
        idade_ticket = df.groupby('Idade_binned')['ValorTotal'].mean().reset_index()
        fig_id = px.bar(idade_ticket, x='Idade_binned', y='ValorTotal')
        st.plotly_chart(fig_id, use_container_width=True)


st.markdown("---")
st.subheader("Tabela de dados")
st.dataframe(df_f.head(200))
