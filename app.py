import os
import pickle
import fitz
import pandas as pd
from flask import Flask, request, render_template, jsonify, send_file
from river import naive_bayes
from scipy.sparse import csr_matrix
import re

# --- CONFIGURAÇÃO DA APLICAÇÃO ---
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
DB_FILE = 'gastos.csv'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
if not os.path.exists(DB_FILE):
    pd.DataFrame(columns=['Arquivo', 'Valor', 'Data', 'Descricao', 'Categoria']).to_csv(DB_FILE, index=False)

# --- CARREGAMENTO DOS MODELOS DE IA ---
try:
    with open('modelo_categoria.pkl', 'rb') as f:
        sklearn_pipeline = pickle.load(f)
    print("Pipeline scikit-learn inicial carregado com sucesso.")
except FileNotFoundError:
    print("ERRO: 'modelo_categoria.pkl' não encontrado. Rode o script de treinamento.")
    exit()
river_classifier = naive_bayes.MultinomialNB()
print("Classificador River preparado para aprendizado online.")

# --- FUNÇÃO AUXILIAR PARA APLICAR REGEX ---
def _find_first_match(patterns, text):
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None

# --- FUNÇÕES DE EXTRAÇÃO DE PDF ---
def extrair_texto_pdf(pdf_path):
    try:
        with fitz.open(pdf_path) as doc:
            return "".join(page.get_text() for page in doc)
    except Exception as e:
        print(f"Erro ao ler o PDF {pdf_path}: {e}")
        return ""

# --- FUNÇÃO DE EXTRAÇÃO DE INFORMAÇÕES ---
def extrair_informacoes_com_regex(texto):
    padroes_data = [
        r"DATA DE EMISSÃO.*?(\d{2}/\d{2}/\d{4})",
        r"Emissão:\s*(\d{2}/\d{2}/\d{4})",
        r"(\d{2}/\d{2}/\d{4})"
    ]
    data_extraida = _find_first_match(padroes_data, texto)
    
    valor_extraido = None
    bloco_calculo = re.search(r"CÁLCULO DO IMPOSTO(.*?)DADOS ADICIONAIS", texto, flags=re.DOTALL | re.IGNORECASE)
    if bloco_calculo:
        padroes_valor_bloco = [
            r"VALOR TOTAL DA NOTA\s*([\d\.]+,\d{2})",
            r"VALOR TOTAL DOS PRODUTOS\s*([\d\.]+,\d{2})",
            r"TOTAL\s*([\d\.]+,\d{2})",
        ]
        valor_extraido = _find_first_match(padroes_valor_bloco, bloco_calculo.group(1))

    if not valor_extraido:
        padroes_valor_genericos = [
            r"TOTAL A PAGAR\s*R?\$\s*([\d\.,]+)",
            r"Valor Total:\s*([\d\.]+,\d{2})",
            r"TOTAL\s*(?:\s*R\$)?\s*([\d\.]+,\d{2})",
            r"VALOR LÍQUIDO.*?([\d\.]+,\d{2})",
            r"R\$\s*([\d\.]+,\d{2})",
        ]
        valor_extraido = _find_first_match(padroes_valor_genericos, texto)

    descricao_limpa = None
    bloco_produtos = re.search(r"DADOS DO PRODUTO/SERVICOS(.*?)CÁLCULO DO ISSQN", texto, flags=re.DOTALL | re.IGNORECASE)
    if bloco_produtos:
        descricoes = re.findall(r"\d+\s+(.*?)\s+\d+", bloco_produtos.group(1))
        if descricoes:
            descricao_limpa = ", ".join([d.strip() for d in descricoes])
    
    if not descricao_limpa:
        padroes_descricao_fornecedor = [
            r"TRANSPORTADOR.*?NOME / RAZÃO SOCIAL\s*(.*?)\s*FRETE POR CONTA",
            r"DESTINATARIO/REMETENTE\s*NOME\s*RAZÃO\s*SOCIAL\s*\n(.*?)\s*ENDEREÇO",
            r"Razão Social\s*(.*?)\s*Endereço:", r"NOME:\s*(.*?)(\n|\r)",
        ]
        descricao_limpa = _find_first_match(padroes_descricao_fornecedor, texto)

    return {
        "data": data_extraida or "Não encontrada",
        "valor": valor_extraido or "Não encontrado",
        "descricao_limpa": descricao_limpa or "Não encontrada"
    }

# --- FUNÇÕES DO MODELO DE IA ---
def prever_categoria(texto):
    return sklearn_pipeline.predict([texto])[0]

def sparse_to_river_dict(sparse_matrix):
    if not isinstance(sparse_matrix, csr_matrix): return {}
    row = sparse_matrix.getrow(0)
    return {int(i): float(v) for i, v in zip(row.indices, row.data)}

# --- ROTAS DA APLICAÇÃO WEB (FLASK) ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400

    resultados = []
    for file in files:
        if file.filename:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            texto = extrair_texto_pdf(filepath)
            info = extrair_informacoes_com_regex(texto)
            
            resultados.append({
                'arquivo': file.filename, 'texto_completo': texto,
                'valor': info['valor'], 'data': info['data'],
                'descricao_exibir': info['descricao_limpa'],
                'categoria_prevista': prever_categoria(texto)
            })
    return jsonify(resultados)

@app.route('/salvar', methods=['POST'])
def salvar_dados():
    try:
        dados_confirmados = request.json['data']
        df_novos = pd.DataFrame(dados_confirmados)
    except (KeyError, TypeError):
        return jsonify({'error': 'JSON inválido ou chave "data" ausente.'}), 400
    
    for _, gasto in df_novos.iterrows():
        descricao_completa = gasto['DescricaoCompleta']
        categoria_correta = gasto['Categoria']
        vetor = sklearn_pipeline.named_steps['vectorizer'].transform([descricao_completa])
        features = sparse_to_river_dict(vetor)
        if features:
            river_classifier.learn_one(x=features, y=categoria_correta)
            print(f"Modelo River aprendeu: '{descricao_completa[:30]}...' é '{categoria_correta}'")

    colunas = ['Arquivo', 'Valor', 'Data', 'Descricao', 'Categoria']
    df_novos[colunas].to_csv(DB_FILE, mode='a', header=False, index=False)
    
    df_total = pd.read_csv(DB_FILE).dropna(subset=['Descricao', 'Categoria'])
    if len(df_total) > 0 and len(df_total) % 5 == 0:
        print(f"Total de {len(df_total)} registros. Re-treinando modelo base...")
        sklearn_pipeline.fit(df_total['Descricao'], df_total['Categoria'])
        with open('modelo_categoria.pkl', 'wb') as f:
            pickle.dump(sklearn_pipeline, f)
        print("Modelo 'modelo_categoria.pkl' atualizado e salvo!")

    return jsonify({'success': 'Dados salvos e modelo atualizado!'})

@app.route('/exportar', methods=['GET'])
def exportar_dados():
    if not os.path.exists(DB_FILE):
        return jsonify({'error': 'Arquivo de dados não encontrado.'}), 404
    try:
        df = pd.read_csv(DB_FILE)
        df['Valor'] = df['Valor'].str.replace('.', '', regex=False).str.replace(',', '.', regex=True).astype(float)
        total_gastos = df['Valor'].sum()
        df.loc[len(df.index)] = {'Descricao' : 'Total Gastos:', 'Categoria': total_gastos}
        
        output_file = "relatorio_de_gastos.xlsx"
        df.to_excel(output_file, index=False)
        return send_file(output_file, as_attachment=True)
    except Exception as e:
        return jsonify({'error': f'Erro ao exportar a planilha: {e}'}), 500
    

@app.route('/resetar', methods=['POST'])
def resetar_dados():
    # --- FUNÇÃO PARA RESETAR A PLANILHA ---
    try:
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)
            print(f"Arquivo '{DB_FILE}' removido com sucesso.")
        
        # Recria o arquivo com os cabeçalhos para que a aplicação continue funcionando
        pd.DataFrame(columns=['Arquivo', 'Valor', 'Data', 'Descricao', 'Categoria']).to_csv(DB_FILE, index=False)
        print(f"Arquivo '{DB_FILE}' recriado com os cabeçalhos.")
        
        return jsonify({'success': 'A planilha foi resetada com sucesso!'})
    except Exception as e:
        print(f"Erro ao tentar resetar a planilha: {e}")
        return jsonify({'error': f'Ocorreu um erro no servidor: {e}'}), 500

# --- INICIALIZAÇÃO DA APLICAÇÃO ---
if __name__ == '__main__':
    app.run(debug=True)