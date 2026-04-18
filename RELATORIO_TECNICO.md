# Relatório Técnico — AgroSmart
## Sistema de Monitoramento Inteligente de Lavoura com IA e Dashboard Analítico

**Projeto:** AgroSmart — Identificação de Doenças em Folhas e Visualização de Dados  
**Instituição:** FIAP  
**Disciplina:** Inteligência Artificial e Visão Computacional  
**Versão:** 2.0 (Fase 1 + Fase 2)  
**Data:** Abril de 2026

---

## 1. Visão Geral do Sistema

O AgroSmart é uma solução completa de tecnologia aplicada ao agronegócio, composta por dois módulos principais:

- **Módulo de Análise (Fase 1):** Pipeline de Visão Computacional com IA para classificar folhas como SAUDÁVEL ou DOENTE em tempo real, via webcam ou imagem estática
- **Módulo de Dashboard (Fase 2):** Painel analítico integrado que organiza, armazena e visualiza os dados coletados, oferecendo métricas e tendências para apoio à decisão do agricultor

Toda a solução é **local e offline**, sem dependência de APIs externas ou servidores em nuvem.

---

## 2. Tecnologias Utilizadas

| Tecnologia | Versão | Finalidade |
|---|---|---|
| Python | 3.x | Linguagem base |
| TensorFlow / Keras | 2.x | CNN de classificação + MobileNetV2 |
| OpenCV (cv2) | 4.x | Segmentação HSV, câmera, morfologia |
| Pillow (PIL) | — | Renderização de frames na GUI |
| Tkinter + ttk | stdlib | Interface gráfica nativa |
| Matplotlib | 3.x | Gráficos embarcados no dashboard |
| Pandas | 2.x | Leitura, escrita e manipulação de CSV |
| NumPy | 1.x | Operações matriciais e pré-processamento |

---

## 3. Arquitetura do Sistema

### 3.1 Pipeline de Detecção de Plantas (4 camadas)

O sistema recusa processar qualquer região que não seja confirmada como planta, evitando falsos positivos com objetos do ambiente:

```
Frame (câmera ou imagem)
        │
        ▼
[1] Segmentação HSV dupla
    - Máscara verde  (Hue 25–95):  plantas saudáveis
    - Máscara marrom (Hue 10–30):  plantas doentes/com necrose
    - Morfologia: CLOSE → OPEN (remove ruído, fecha buracos)
        │ nenhum contorno → rejeita
        ▼
[2] Validação de região
    - Área mínima: max(8000px, 3% do frame)
    - Densidade de pixels vegetais ≥ 25% da bbox
    - Proporção (aspecto): 0.15 < w/h < 6
        │ falhou → rejeita
        ▼
[3] Textura orgânica (Laplaciano)
    - Variância do Laplaciano > 80
    - Folhas possuem veios e bordas → alta variância
    - Objetos lisos (paredes, roupas) são rejeitados
        │ falhou → rejeita
        ▼
[4] Verificação semântica — MobileNetV2 (ImageNet)
    - Top-5 predições verificadas por palavras-chave vegetais
    - Confirma: leaf, plant, flower, fern, grass, corn, etc.
        │ não é planta → rejeita
        ▼
   Recorte validado → Classificador AgroSmart
```

### 3.2 Pipeline de Classificação (Sistema Híbrido)

```
Recorte da planta validada
        │
        ▼
CLAHE (normalização de iluminação)
    - Equalização adaptativa de histograma em espaço LAB
    - Estabiliza resultados em ambientes com luz variável
        │
        ▼
CNN AgroSmart (modelo_agrosmart.h5)
    - Entrada: 128×128 RGB normalizado [0,1]
    - Saída: score 0.0–1.0 (0=doente, 1=saudável)
    - Peso no score final: 65%
        │
        ▼
Heurística de manchas
    - Detecta marrom escuro (necrose/ferrugem): Hue 0–25
    - Detecta amarelo claro (clorose): Hue 20–35, V>150
    - Proporção de pixels doentes no recorte × 3.0
    - Peso no score final: 35%
        │
        ▼
Suavização temporal (buffer de 5 frames)
    - Média das últimas 5 predições
    - Elimina flickering entre frames consecutivos
        │
        ▼
score < 0.40 → SAUDÁVEL   (bounding box verde)
score ≥ 0.40 → DOENTE     (bounding box vermelha)
```

### 3.3 Modelo CNN — Arquitetura

Rede Neural Convolucional treinada do zero com 15.269 imagens:

```
Input (128×128×3)
  → Conv2D(32, 3×3, ReLU) → MaxPooling(2×2)
  → Conv2D(64, 3×3, ReLU) → MaxPooling(2×2)
  → Conv2D(128, 3×3, ReLU) → MaxPooling(2×2)
  → Flatten
  → Dense(128, ReLU) → Dropout(0.5)
  → Dense(1, Sigmoid)   ← saída binária
```

**Treinamento:**
- Épocas: 10 | Batch: 16 | Otimizador: Adam | Loss: Binary Crossentropy
- Data augmentation: rotação 20°, zoom 15%, flip horizontal
- Split: 70% treino / 20% validação / 10% teste
- Dataset: 4 bases do Roboflow (~10.687 doentes, ~4.582 saudáveis)

---

## 4. Módulo de Dashboard (Fase 2)

### 4.1 Estrutura de Dados

O sistema integra duas fontes de dados via CSV (separador `;`):

| Arquivo | Conteúdo |
|---|---|
| `logs/dados_agrosmart.csv` | Registros reais gerados pela câmera |
| `logs/dados_simulados.csv` | 102 registros simulados (Jan–Abr 2026) |

**Campos do CSV:**
```
Nome da Imagem | Data/Hora | Categoria Detectada | Acurácia (Confiança) | Tipo de Anomalia | Localidade | Cultura
```

**Dados simulados cobrem:**
- 4 localidades: Fazenda Norte, Fazenda Sul, Sítio Leste, Fazenda Oeste
- 4 culturas: Soja, Milho, Café, Trigo
- 5 tipos de anomalia: Mancha Foliar, Ferrugem, Míldio, Antracnose, Nenhuma

### 4.2 Métricas e Visualizações

| Métrica | Tipo de Gráfico | Insight para o Agricultor |
|---|---|---|
| Total de imagens analisadas | KPI card | Volume de monitoramento |
| % saudável vs doente | Pizza (donut) | Estado geral da lavoura |
| Frequência por anomalia | Barras horizontais | Praga predominante |
| Tendência mensal | Linha com área | Evolução ao longo do tempo |
| Saúde por localidade | Barras agrupadas | Área mais afetada |
| Confiança média da IA | KPI card | Qualidade das detecções |

### 4.3 Interface — Organização em Abas

```
┌─────────────────────────────────────────────────────┐
│  🌱 AgroSmart FIAP — Monitoramento Inteligente      │
├──────────────┬──────────────────┬───────────────────┤
│ 📊 Dashboard │ 📷 Câmera/Análise│ 📋 Registros      │
├──────────────┴──────────────────┴───────────────────┤
│                                                      │
│  [KPIs: Total | Saudáveis | Doentes | Confiança]    │
│                                                      │
│  [Pizza]  [Anomalias]  [Por Localidade]             │
│                                                      │
│  [Tendência Mensal — linha temporal]                │
└──────────────────────────────────────────────────────┘
```

---

## 5. Aplicabilidade e Impacto

### 5.1 Apoio à Decisão do Agricultor

O dashboard responde perguntas práticas do campo:

- **"Qual área da fazenda está mais doente?"** → Gráfico por localidade
- **"Que praga está predominando neste mês?"** → Ranking de anomalias
- **"A situação está piorando ou melhorando?"** → Tendência mensal
- **"Quanto da minha lavoura está comprometida?"** → KPI de % doentes

### 5.2 Escalabilidade

- O fluxo de dados é baseado em CSV simples — facilmente substituível por banco de dados (SQLite, PostgreSQL) ou integração com Google Sheets
- O modelo pode ser retreinado com novos dados inserindo imagens nas pastas `data/dataset/train/`
- A arquitetura modular permite adicionar novas culturas e tipos de anomalia sem reescrever o sistema

### 5.3 Acessibilidade

- Roda em qualquer computador com Python e Anaconda, sem GPU
- Interface gráfica nativa (Tkinter), sem necessidade de browser
- Processamento 100% local — dados do agricultor nunca saem do computador

---

## 6. Execução

```bash
# Com Anaconda (recomendado — sem necessidade de venv)
pip install -r requirements.txt
python app.py
```

Na primeira execução, o MobileNetV2 (~14 MB) é baixado automaticamente do TensorFlow Hub.

---

## 7. Limitações e Trabalhos Futuros

| Limitação Atual | Melhoria Proposta |
|---|---|
| Classificação binária (saudável/doente) | Classificação multi-classe por tipo de doença |
| MobileNet rodando a cada frame | Cache a cada N frames para maior fluidez |
| Dataset desbalanceado (65% doente) | Coletar mais imagens de plantas saudáveis |
| CSV como banco de dados | Migrar para SQLite com histórico persistente |
| Sem identificação da cultura | Adicionar seletor de cultura antes da análise |
