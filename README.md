# AgroSmart — Sistema de Monitoramento Inteligente de Lavoura 🌱

Sistema de Visão Computacional e IA desenvolvido para auxiliar agricultores na identificação de folhas **SAUDÁVEIS** e **DOENTES**, com dashboard interativo de análise de dados.

Projeto desenvolvido para a disciplina de Inteligência Artificial e Visão Computacional — **FIAP**.

---

## Funcionalidades

### Fase 1 — Análise de Folhas com IA
- 🧠 **CNN Local (TensorFlow/Keras):** Rede neural treinada com +15.000 imagens, roda offline na CPU
- 📷 **Análise por Webcam:** Captura em tempo real com detecção automática de plantas
- 🖼️ **Upload de Imagem:** Análise de imagens estáticas `.jpg`, `.png`, `.bmp`
- 🔍 **Pipeline de detecção em 4 camadas:**
  1. Segmentação HSV (verde + marrom/amarelo)
  2. Validação de densidade e proporção da região
  3. Verificação de textura orgânica (Laplaciano)
  4. Confirmação por MobileNetV2 pré-treinado no ImageNet
- 🤖 **Sistema híbrido:** CNN AgroSmart (65%) + heurística de manchas (35%) com suavização temporal
- ⚡ **CLAHE:** Normalização de iluminação antes da inferência
- 💾 **Exportação CSV:** Log automático de diagnósticos com data/hora e confiança

### Fase 2 — Dashboard de Monitoramento
- 📊 **Painel interativo integrado ao Tkinter** com 3 abas:
  - **Dashboard:** KPIs, gráfico de pizza, anomalias, tendência mensal, saúde por localidade
  - **Câmera / Análise:** Interface de análise em tempo real
  - **Registros:** Tabela completa com todos os dados coletados
- 📁 **Integração com CSV:** Lê dados reais (`dados_agrosmart.csv`) e simulados (`dados_simulados.csv`)
- 🔄 **Atualização sob demanda:** Botão de refresh no dashboard

---

## Estrutura do Projeto

```
agrosmart/
├── app.py                      # Aplicação principal (dashboard + câmera)
├── train.py                    # Script de treinamento do modelo CNN
├── requirements.txt            # Dependências Python
├── README.md
├── RELATORIO_TECNICO.md
│
├── data.zip/
│   └── dataset/                # Imagens de treino/validação/teste
│       ├── train/
│       │   ├── saudavel/       # 2.518 imagens
│       │   └── doente/         # 8.169 imagens
│       ├── valid/
│       └── test/
│
├── logs/
│   ├── dados_agrosmart.csv     # Registros reais gerados pela câmera
│   └── dados_simulados.csv     # Dados simulados para demonstração (Fase 2)
│
└── models/
    └── modelo_agrosmart.h5     # Modelo CNN treinado (~37 MB)
```

---

## Como Instalar e Rodar

### Pré-requisito: Anaconda (recomendado)

O projeto usa Python 3.x. Com Anaconda instalado, não é necessário criar venv.

**1. Instale as dependências:**
```bash
pip install -r requirements.txt
```

**2. Execute:**
```bash
python app.py
```

O MobileNetV2 (~14 MB) é baixado automaticamente na primeira execução.

---

## Guia da Interface

A aplicação abre diretamente no **Dashboard**:

| Aba | Descrição |
|-----|-----------|
| 📊 Dashboard | KPIs gerais, gráficos interativos, tendência mensal |
| 📷 Câmera / Análise | Análise ao vivo por webcam ou upload de imagem |
| 📋 Registros | Tabela com todos os diagnósticos registrados |

### Aba Câmera / Análise
- **📷 Ligar Câmera:** Inicia captura ao vivo. Aponte uma folha para a câmera — o sistema desenha uma bounding box verde (saudável) ou vermelha (doente) somente quando confirma que o objeto é uma planta
- **🖼 Enviar Imagem:** Carrega uma imagem do disco para análise
- **💾 Salvar CSV:** Salva o diagnóstico atual no log (habilitado após uma detecção)

---

## Dependências

```
tensorflow       # CNN + MobileNetV2
opencv-python    # Visão computacional e câmera
pandas           # Manipulação de dados CSV
numpy            # Operações matriciais
Pillow           # Renderização de imagens na GUI
matplotlib       # Gráficos do dashboard
```
