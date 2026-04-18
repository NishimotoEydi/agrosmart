import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from datetime import datetime
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec

MODEL_PATH = "models/modelo_agrosmart.h5"
CSV_PATH   = "logs/dados_agrosmart.csv"
CSV_SIM    = "logs/dados_simulados.csv"
IMG_WIDTH  = 128
IMG_HEIGHT = 128

try:
    modelo = tf.keras.models.load_model(MODEL_PATH)
except Exception:
    exit()

# MobileNetV2 pré-treinado no ImageNet — usado como "porteiro" de plantas
# Índices ImageNet de folhas/plantas/flores conhecidos
_PLANT_IDXS = set([
    # folhas, samambaias, flores, cogumelos, vegetação
    *range(984, 1000),  # flores diversas
    *range(944, 960),   # plantas/cogumelos
    992, 993, 994, 995, 996, 997, 998, 999,
    # árvores e arbustos
    340, 341, 984, 985, 986, 987, 988, 989, 990, 991,
])
try:
    _mobilenet = tf.keras.applications.MobileNetV2(
        weights="imagenet", include_top=True, input_shape=(224, 224, 3)
    )
    _mobilenet_decode = tf.keras.applications.mobilenet_v2.decode_predictions
    _mobilenet_pre    = tf.keras.applications.mobilenet_v2.preprocess_input
    _USE_MOBILENET = True
except Exception:
    _USE_MOBILENET = False


def _e_planta_mobilenet(frame_bgr) -> tuple[bool, float]:
    """Retorna (é_planta, confiança) usando MobileNetV2."""
    if not _USE_MOBILENET:
        return True, 1.0
    img = cv2.resize(frame_bgr, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32")
    arr = _mobilenet_pre(np.expand_dims(img, 0))
    preds = _mobilenet(arr, training=False).numpy()
    top5  = _mobilenet_decode(preds, top=5)[0]
    # Verifica se algum dos top-5 é categoria vegetal
    for _, label, conf in top5:
        label_l = label.lower()
        if any(k in label_l for k in (
            "leaf", "plant", "flower", "fern", "moss", "vegetable",
            "herb", "tree", "shrub", "fungus", "mushroom", "grass",
            "daisy", "rose", "tulip", "corn", "cucumber", "cabbage",
        )):
            return True, float(conf)
    # Também aceita pelo índice
    top_idx = int(np.argmax(preds[0]))
    if top_idx in _PLANT_IDXS:
        return True, float(preds[0][top_idx])
    return False, 0.0

# ── Paleta ───────────────────────────────────────────────────────────────────
BG      = "#1a2e1a"
PAINEL  = "#243324"
VERDE   = "#4caf50"
VERDE2  = "#81c784"
VERM    = "#e53935"
AMAR    = "#fdd835"
TEXTO   = "#e8f5e9"
CINZA   = "#90a4ae"
AZUL    = "#1565c0"
ROXO    = "#6a1b9a"


def carregar_dados():
    frames = []
    for path in [CSV_SIM, CSV_PATH]:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
                frames.append(df)
            except Exception:
                pass
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df.columns = df.columns.str.strip()
    for col in ["Tipo de Anomalia", "Localidade", "Cultura"]:
        if col not in df.columns:
            df[col] = "Não informado"
    df["Data/Hora"] = pd.to_datetime(df["Data/Hora"], errors="coerce")
    df["Acurácia (Confiança)"] = (
        df["Acurácia (Confiança)"]
        .astype(str).str.replace("%", "", regex=False).str.strip().astype(float)
    )
    df["Mês"] = df["Data/Hora"].dt.to_period("M").astype(str)
    return df.dropna(subset=["Data/Hora"])


class AgroSmartApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AgroSmart FIAP — Monitoramento de Lavoura")
        self.root.geometry("1100x750")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)

        self.camera = None
        self.is_camera_on = False
        self.categoria_atual = ""
        self.confianca_atual = 0.0
        # Suavização: média das últimas N predições para evitar flickering
        self._historico_scores = []
        self._N_FRAMES = 5

        self._build_header()
        self._build_notebook()

    # ── Cabeçalho ─────────────────────────────────────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self.root, bg=PAINEL, pady=10)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🌱  AgroSmart FIAP — Monitoramento Inteligente de Lavoura",
                 font=("Arial", 15, "bold"), bg=PAINEL, fg=VERDE).pack(side="left", padx=20)
        tk.Label(hdr, text=datetime.now().strftime("%d/%m/%Y"),
                 font=("Arial", 10), bg=PAINEL, fg=CINZA).pack(side="right", padx=20)

    # ── Notebook principal ────────────────────────────────────────────────────
    def _build_notebook(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook",      background=BG, borderwidth=0)
        style.configure("TNotebook.Tab",  background=PAINEL, foreground=TEXTO,
                        padding=[18, 8], font=("Arial", 10, "bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", VERDE)],
                  foreground=[("selected", "white")])

        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True, padx=10, pady=8)

        # Abas
        self.aba_dash    = tk.Frame(self.nb, bg=BG)
        self.aba_camera  = tk.Frame(self.nb, bg=BG)
        self.aba_tabela  = tk.Frame(self.nb, bg=BG)

        self.nb.add(self.aba_dash,   text="📊  Dashboard")
        self.nb.add(self.aba_camera, text="📷  Câmera / Análise")
        self.nb.add(self.aba_tabela, text="📋  Registros")

        self._build_dashboard()
        self._build_camera_tab()
        self._build_tabela()

        # Atualiza câmera só quando a aba estiver visível
        self.nb.bind("<<NotebookTabChanged>>", self._on_tab_change)
        self.update_webcam()

    # ══════════════════════════════════════════════════════════════════════════
    # ABA DASHBOARD
    # ══════════════════════════════════════════════════════════════════════════
    def _build_dashboard(self):
        for w in self.aba_dash.winfo_children():
            w.destroy()

        df = carregar_dados()

        # KPIs
        frame_kpi = tk.Frame(self.aba_dash, bg=BG)
        frame_kpi.pack(fill="x", padx=12, pady=(10, 6))

        if df.empty:
            tk.Label(self.aba_dash, text="Nenhum dado encontrado.",
                     bg=BG, fg=TEXTO, font=("Arial", 13)).pack(pady=40)
            return

        total     = len(df)
        saudaveis = (df["Categoria Detectada"] == "SAUDÁVEL").sum()
        doentes   = (df["Categoria Detectada"] == "DOENTE").sum()
        pct_d     = doentes / total * 100 if total else 0
        conf_med  = df["Acurácia (Confiança)"].mean()

        kpis = [
            ("📷 Imagens\nAnalisadas", str(total),         PAINEL),
            ("✅ Saudáveis",           str(saudaveis),      "#1b5e20"),
            (f"⚠️ Doentes\n({pct_d:.1f}%)", str(doentes),  "#b71c1c"),
            ("🎯 Confiança\nMédia IA", f"{conf_med:.1f}%", "#0d47a1"),
        ]
        for i, (lbl, val, cor) in enumerate(kpis):
            card = tk.Frame(frame_kpi, bg=cor, padx=18, pady=10)
            card.grid(row=0, column=i, padx=8, sticky="ew")
            frame_kpi.grid_columnconfigure(i, weight=1)
            tk.Label(card, text=val, font=("Arial", 20, "bold"),
                     bg=cor, fg="white").pack()
            tk.Label(card, text=lbl, font=("Arial", 8),
                     bg=cor, fg="#cccccc").pack()

        # Botão atualizar
        tk.Button(self.aba_dash, text="🔄  Atualizar Dashboard",
                  bg=VERDE, fg="white", font=("Arial", 9, "bold"),
                  relief="flat", padx=10, pady=4,
                  command=self._build_dashboard).pack(anchor="e", padx=14, pady=(0, 4))

        # Gráficos — linha 1: pizza + anomalias + localidade
        fig1 = Figure(figsize=(11, 3.6), facecolor=BG)
        gs1  = gridspec.GridSpec(1, 3, figure=fig1, wspace=0.42,
                                 left=0.05, right=0.97, top=0.88, bottom=0.18)

        # Pizza
        ax_pizza = fig1.add_subplot(gs1[0])
        ax_pizza.set_facecolor(BG)
        ax_pizza.pie(
            [saudaveis, doentes],
            labels=["Saudável", "Doente"],
            colors=[VERDE, VERM], autopct="%1.1f%%",
            startangle=90, wedgeprops=dict(width=0.55),
            textprops=dict(color=TEXTO, fontsize=9),
        )
        ax_pizza.set_title("Saúde das Folhas", color=TEXTO, fontsize=10, pad=8)

        # Anomalias
        ax_ano = fig1.add_subplot(gs1[1])
        ax_ano.set_facecolor(PAINEL)
        anomalias = (
            df[df["Categoria Detectada"] == "DOENTE"]
            .groupby("Tipo de Anomalia").size().sort_values()
        )
        if not anomalias.empty:
            bars = ax_ano.barh(anomalias.index, anomalias.values,
                               color=AMAR, edgecolor="none")
            for b in bars:
                ax_ano.text(b.get_width() + 0.1, b.get_y() + b.get_height()/2,
                            str(int(b.get_width())), va="center",
                            color=TEXTO, fontsize=8)
        ax_ano.set_title("Anomalias Detectadas", color=TEXTO, fontsize=10, pad=8)
        ax_ano.tick_params(colors=CINZA, labelsize=7)
        ax_ano.spines[:].set_color(PAINEL)

        # Por localidade
        ax_loc = fig1.add_subplot(gs1[2])
        ax_loc.set_facecolor(PAINEL)
        por_local = (
            df.groupby(["Localidade", "Categoria Detectada"]).size()
            .unstack(fill_value=0)
        )
        if not por_local.empty:
            xs = range(len(por_local))
            w  = 0.35
            cores_map = {"SAUDÁVEL": VERDE, "DOENTE": VERM}
            for i, cat in enumerate(por_local.columns):
                offset = (i - len(por_local.columns)/2 + 0.5) * w
                ax_loc.bar([x + offset for x in xs], por_local[cat], width=w,
                           label=cat, color=cores_map.get(cat, CINZA))
            ax_loc.set_xticks(list(xs))
            ax_loc.set_xticklabels(por_local.index, rotation=18, ha="right",
                                   fontsize=6, color=CINZA)
            ax_loc.legend(fontsize=7, labelcolor=TEXTO,
                          facecolor=PAINEL, edgecolor="none")
        ax_loc.set_title("Por Localidade", color=TEXTO, fontsize=10, pad=8)
        ax_loc.tick_params(colors=CINZA, labelsize=7)
        ax_loc.spines[:].set_color(PAINEL)

        cv1 = FigureCanvasTkAgg(fig1, master=self.aba_dash)
        cv1.draw()
        cv1.get_tk_widget().pack(fill="x", padx=10)

        # Gráfico — linha 2: tendência mensal
        tend = (
            df.groupby(["Mês", "Categoria Detectada"]).size()
            .unstack(fill_value=0).reset_index().sort_values("Mês")
        )
        fig2 = Figure(figsize=(11, 2.6), facecolor=BG)
        ax_t = fig2.add_subplot(111)
        ax_t.set_facecolor(PAINEL)
        if "SAUDÁVEL" in tend.columns:
            ax_t.plot(tend["Mês"], tend["SAUDÁVEL"], marker="o",
                      color=VERDE, linewidth=2, label="Saudável")
            ax_t.fill_between(tend["Mês"], tend["SAUDÁVEL"],
                              alpha=0.12, color=VERDE)
        if "DOENTE" in tend.columns:
            ax_t.plot(tend["Mês"], tend["DOENTE"], marker="o",
                      color=VERM, linewidth=2, label="Doente")
            ax_t.fill_between(tend["Mês"], tend["DOENTE"],
                              alpha=0.12, color=VERM)
        ax_t.set_title("Tendência Mensal de Detecções", color=TEXTO, fontsize=10)
        ax_t.tick_params(colors=CINZA, labelsize=7)
        ax_t.spines[:].set_color(PAINEL)
        ax_t.legend(fontsize=8, labelcolor=TEXTO,
                    facecolor=PAINEL, edgecolor="none")
        import matplotlib.pyplot as plt
        plt.setp(ax_t.get_xticklabels(), rotation=25, ha="right", color=CINZA)
        fig2.tight_layout(pad=0.8)

        cv2 = FigureCanvasTkAgg(fig2, master=self.aba_dash)
        cv2.draw()
        cv2.get_tk_widget().pack(fill="x", padx=10, pady=(0, 8))

    # ══════════════════════════════════════════════════════════════════════════
    # ABA CÂMERA / ANÁLISE
    # ══════════════════════════════════════════════════════════════════════════
    def _build_camera_tab(self):
        tk.Label(self.aba_camera,
                 text="Análise em Tempo Real de Folhas",
                 font=("Arial", 13, "bold"), bg=BG, fg=VERDE).pack(pady=(10, 4))

        self.canvas = tk.Canvas(self.aba_camera, width=640, height=440,
                                bg="black", highlightthickness=2,
                                highlightbackground=VERDE)
        self.canvas.pack()

        self.lbl_resultado = tk.Label(self.aba_camera,
                                      text="IA: Aguardando imagem...",
                                      font=("Arial", 11, "bold"),
                                      bg=BG, fg=CINZA)
        self.lbl_resultado.pack(pady=6)

        btn_row = tk.Frame(self.aba_camera, bg=BG)
        btn_row.pack(pady=4)

        estilo = dict(font=("Arial", 10, "bold"), relief="flat",
                      padx=14, pady=7, cursor="hand2")

        self.btn_camera = tk.Button(btn_row, text="📷  Ligar Câmera",
                                    bg=VERDE, fg="white",
                                    command=self.toggle_camera, **estilo)
        self.btn_camera.grid(row=0, column=0, padx=8)

        tk.Button(btn_row, text="🖼  Enviar Imagem",
                  bg=AZUL, fg="white",
                  command=self.upload_image, **estilo).grid(row=0, column=1, padx=8)

        self.btn_salvar = tk.Button(btn_row, text="💾  Salvar CSV",
                                    bg="#5d4037", fg="white", state=tk.DISABLED,
                                    command=self.salvar_csv, **estilo)
        self.btn_salvar.grid(row=0, column=2, padx=8)

    # ══════════════════════════════════════════════════════════════════════════
    # ABA TABELA
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tabela(self):
        for w in self.aba_tabela.winfo_children():
            w.destroy()

        tk.Label(self.aba_tabela, text="Registros Coletados",
                 font=("Arial", 12, "bold"), bg=BG, fg=VERDE).pack(pady=(10, 4))

        tk.Button(self.aba_tabela, text="🔄  Atualizar",
                  bg=VERDE, fg="white", font=("Arial", 9, "bold"),
                  relief="flat", padx=10, pady=3,
                  command=self._build_tabela).pack(anchor="e", padx=14, pady=(0, 4))

        df = carregar_dados()
        if df.empty:
            tk.Label(self.aba_tabela, text="Nenhum dado encontrado.",
                     bg=BG, fg=TEXTO, font=("Arial", 12)).pack(pady=30)
            return

        colunas = ["Nome da Imagem", "Data/Hora", "Categoria Detectada",
                   "Acurácia (Confiança)", "Tipo de Anomalia", "Localidade", "Cultura"]
        colunas_ex = [c for c in colunas if c in df.columns]

        style = ttk.Style()
        style.configure("Treeview",
                        background=PAINEL, foreground=TEXTO,
                        rowheight=24, fieldbackground=PAINEL,
                        font=("Arial", 9))
        style.configure("Treeview.Heading",
                        background=VERDE, foreground="white",
                        font=("Arial", 9, "bold"))
        style.map("Treeview", background=[("selected", VERDE2)])

        frame_t = tk.Frame(self.aba_tabela, bg=BG)
        frame_t.pack(fill="both", expand=True, padx=10, pady=4)

        vsb = ttk.Scrollbar(frame_t, orient="vertical")
        hsb = ttk.Scrollbar(frame_t, orient="horizontal")
        tree = ttk.Treeview(frame_t, columns=colunas_ex, show="headings",
                            yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.config(command=tree.yview)
        hsb.config(command=tree.xview)

        for col in colunas_ex:
            tree.heading(col, text=col)
            tree.column(col, width=135, anchor="center")

        df_ord = df[colunas_ex].sort_values("Data/Hora", ascending=False)
        for _, row in df_ord.iterrows():
            tag = "doente" if row.get("Categoria Detectada") == "DOENTE" else "saudavel"
            tree.insert("", "end", values=[str(v) for v in row.values], tags=(tag,))

        tree.tag_configure("doente",   background="#3b1a1a", foreground="#ff8a80")
        tree.tag_configure("saudavel", background="#1a3b1a", foreground="#a5d6a7")

        vsb.pack(side="right",  fill="y")
        hsb.pack(side="bottom", fill="x")
        tree.pack(fill="both", expand=True)

        tk.Label(frame_t, text=f"Total: {len(df_ord)} registros",
                 bg=BG, fg=CINZA, font=("Arial", 9)).pack(anchor="e", pady=3)

    # ══════════════════════════════════════════════════════════════════════════
    # LÓGICA DE ANÁLISE (inalterada)
    # ══════════════════════════════════════════════════════════════════════════
    def _on_tab_change(self, event):
        # Para câmera se sair da aba de câmera
        tab = self.nb.index(self.nb.select())
        if tab != 1 and self.is_camera_on:
            self.toggle_camera()

    def analisar_planta(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask_verde  = cv2.inRange(hsv, np.array([25, 40, 40]),  np.array([95, 255, 255]))
        mask_doente = cv2.inRange(hsv, np.array([10, 40, 40]),  np.array([30, 255, 200]))
        mask = cv2.bitwise_or(mask_verde, mask_doente)

        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

        contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contornos:
            return None, None

        AREA_MIN   = max(8000, frame.shape[0] * frame.shape[1] * 0.03)
        candidatos = [c for c in contornos if cv2.contourArea(c) > AREA_MIN]
        if not candidatos:
            return None, None

        maior = max(candidatos, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(maior)

        # Densidade de pixels vegetais na bbox
        roi_mask  = mask[y:y+h, x:x+w]
        densidade = np.count_nonzero(roi_mask) / (w * h)
        if densidade < 0.25:
            return None, None

        # Proporção implausível para folha
        aspecto = w / h if h > 0 else 0
        if aspecto > 6 or aspecto < 0.15:
            return None, None

        # Textura orgânica: folhas têm veios → variância do Laplaciano alta
        gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        textura = cv2.Laplacian(gray, cv2.CV_64F).var()
        if textura < 80:          # objeto liso/sólido (parede, tecido plano)
            return None, None

        # Porteiro MobileNet: confirma que é vegetação
        recorte = frame[y:y+h, x:x+w]
        e_planta, _ = _e_planta_mobilenet(recorte)
        if not e_planta:
            return None, None

        return (x, y, w, h), recorte

    def classificar_imagem(self, frame):
        # CLAHE: normaliza iluminação antes de passar pela IA
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        frame_norm = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        img = cv2.resize(frame_norm, (IMG_WIDTH, IMG_HEIGHT))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        arr = np.expand_dims(np.array(img) / 255.0, axis=0)
        previsao_ia = modelo.predict(arr, verbose=0)[0][0]

        # Heurística: detecta manchas de doença (marrom, amarelo, preto)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Marrom/amarelo escuro (necrose, ferrugem)
        m1 = cv2.inRange(hsv, np.array([0,  40, 20]), np.array([25, 255, 160]))
        # Amarelo claro (clorose)
        m2 = cv2.inRange(hsv, np.array([20, 40, 150]), np.array([35, 255, 255]))
        mask_spot = cv2.bitwise_or(m1, m2)

        total_px  = frame.shape[0] * frame.shape[1]
        spot_ratio = np.count_nonzero(mask_spot) / total_px  # 0..1

        # IA tem peso maior (treinada em 15k imagens); heurística como reforço
        score_bruto = (previsao_ia * 0.65) + (min(spot_ratio * 3.0, 1.0) * 0.35)

        # Suavização temporal: média das últimas N predições
        self._historico_scores.append(score_bruto)
        if len(self._historico_scores) > self._N_FRAMES:
            self._historico_scores.pop(0)
        score = float(np.mean(self._historico_scores))

        if score < 0.40:
            confianca = (1.0 - score) * 100
            return "SAUDÁVEL", confianca, VERDE, (0, 200, 0), previsao_ia, spot_ratio
        else:
            confianca = score * 100
            return "DOENTE", confianca, VERM, (0, 0, 220), previsao_ia, spot_ratio

    def desenhar_frame(self, frame):
        fd = frame.copy()
        bbox, recorte = self.analisar_planta(fd)
        if recorte is not None:
            self.categoria_atual, self.confianca_atual, cor_tk, bgr, raw_ia, raw_h = \
                self.classificar_imagem(recorte)
            x, y, w, h = bbox
            cv2.rectangle(fd, (x, y), (x+w, y+h), bgr, 3)
            cv2.rectangle(fd, (x, y-30), (x+w, y), bgr, -1)
            txt = f"{self.categoria_atual} ({self.confianca_atual:.1f}%) | AI:{raw_ia:.2f} H:{raw_h:.2f}"
            cv2.putText(fd, txt, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            self.lbl_resultado.config(text=f"SISTEMA HÍBRIDO: {txt}", fg=cor_tk)
            self.btn_salvar.config(state=tk.NORMAL)
        else:
            self.categoria_atual = ""
            self.confianca_atual = 0.0
            self.lbl_resultado.config(text="IA: Nenhuma planta detectada", fg=AMAR)
            self.btn_salvar.config(state=tk.DISABLED)

        rgb = cv2.cvtColor(fd, cv2.COLOR_BGR2RGB)
        h_img, w_img = rgb.shape[:2]
        razao = min(640/w_img, 440/h_img)
        nw, nh = int(w_img*razao), int(h_img*razao)
        fundo = Image.new("RGB", (640, 440), (0, 0, 0))
        fundo.paste(Image.fromarray(cv2.resize(rgb, (nw, nh))),
                    ((640-nw)//2, (440-nh)//2))
        self.photo = ImageTk.PhotoImage(image=fundo)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def toggle_camera(self):
        if self.is_camera_on:
            self.is_camera_on = False
            self.btn_camera.config(text="📷  Ligar Câmera")
            if self.camera:
                self.camera.release()
            self.canvas.delete("all")
            self.lbl_resultado.config(text="IA: Câmera Desligada", fg=CINZA)
            self.btn_salvar.config(state=tk.DISABLED)
        else:
            self.camera = cv2.VideoCapture(0)
            if self.camera.isOpened():
                self.is_camera_on = True
                self.btn_camera.config(text="🔴  Desligar Câmera")
            else:
                messagebox.showerror("Erro", "Não foi possível acessar a Webcam.")

    def update_webcam(self):
        if self.is_camera_on and self.camera and self.camera.isOpened():
            ok, frame = self.camera.read()
            if ok:
                self.desenhar_frame(frame)
        self.root.after(15, self.update_webcam)

    def upload_image(self):
        if self.is_camera_on:
            self.toggle_camera()
        path = filedialog.askopenfilename(
            title="Selecione uma Imagem",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp")]
        )
        if path:
            frame = cv2.imread(path)
            if frame is not None:
                self.desenhar_frame(frame)
            else:
                messagebox.showerror("Erro", "Não foi possível carregar a imagem.")

    def salvar_csv(self):
        if not self.categoria_atual:
            return
        nome = f"captura_{datetime.now().strftime('%H%M%S')}"
        novo = pd.DataFrame([{
            "Nome da Imagem":       nome,
            "Data/Hora":            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Categoria Detectada":  self.categoria_atual,
            "Acurácia (Confiança)": f"{self.confianca_atual:.2f}%",
            "Tipo de Anomalia":     "Não informado",
            "Localidade":           "Não informado",
            "Cultura":              "Não informado",
        }])
        os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
        if not os.path.exists(CSV_PATH):
            novo.to_csv(CSV_PATH, index=False, sep=";")
        else:
            novo.to_csv(CSV_PATH, mode="a", header=False, index=False, sep=";")
        messagebox.showinfo("Sucesso", f"Salvo!\nImagem: {nome}\nClasse: {self.categoria_atual}")


if __name__ == "__main__":
    root = tk.Tk()
    AgroSmartApp(root)
    root.mainloop()
