from fastapi import FastAPI, Query, HTTPException
from transformers import pipeline
import json
import os
from typing import List, Dict
from functools import lru_cache

from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


dir_atual = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.normpath(os.path.join(
    dir_atual, "../comentarios_organizados.json"))

try:
    with open(json_path, "r", encoding="utf-8") as f:
        comentarios_raw = json.load(f)
    print(f"DEBUG: Carregados {len(comentarios_raw)} comentários")
    # Ordenar uma vez na inicialização para melhor performance
    comentarios_raw = sorted(
        comentarios_raw,
        key=lambda c: c.get("data_criacao", ""),
        reverse=True
    )
except FileNotFoundError:
    print(f"ERROR: Arquivo não encontrado em {json_path}")
    comentarios_raw = []


# Cache do modelo para evitar recarregar
@lru_cache(maxsize=1)
def get_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        return_all_scores=False,
        device=-1  # Força CPU para economizar memória
    )


@app.get("/debug/keys/")
def debug_keys():
    primeiro = comentarios_raw[0] if comentarios_raw else {}
    return {"keys_do_primeiro_comentario": list(primeiro.keys())}


@app.get("/comentarios/")
def listar_comentarios(page: int = Query(1, ge=1), limit: int = Query(30, ge=1, le=100)):
    # Dados já ordenados na inicialização
    start = (page - 1) * limit
    end = start + limit
    comentarios = []

    sentiment_model = get_sentiment_model()

    for c in comentarios_raw[start:end]:
        texto = c.get("corpo", "").strip()
        if not texto:
            continue
        analysis = sentiment_model(texto, truncation=True)
        pred = analysis[0]
        label = pred.get("label", "").lower()
        if label == "positive":
            sentimento = "Positivo"
        elif label == "negative":
            sentimento = "Negativo"
        else:
            sentimento = "Neutro"
        comentarios.append({
            "id": c.get("id"),
            "id_autor": c.get("id_autor"),
            "comentario": texto,
            "data_criacao": c.get("data_criacao"),
            "sentimento": sentimento
        })
    return {
        "comentarios": comentarios,
        "total": len(comentarios_raw)
    }


@app.get("/debug/")
def debug():
    return {"total_comentarios": len(comentarios_raw)}


@app.get("/propostas/")
def listar_propostas():
    ids_unicos = sorted({c["id_comentavel_raiz"] for c in comentarios_raw})
    return [{"id": i, "titulo": f"Proposta {i}"} for i in ids_unicos]


@app.get("/sentimentos/")
def analisar_sentimentos(id_proposta: str = Query(..., alias="id")):
    filtrados = [
        c for c in comentarios_raw
        if str(c.get("id_comentavel_raiz")) == id_proposta
    ]
    if not filtrados:
        raise HTTPException(
            status_code=404,
            detail=f"Nenhum comentário encontrado para id {id_proposta}"
        )

    resultados = []
    sentiment_model = get_sentiment_model()

    for c in filtrados:
        texto = c.get("corpo", "").strip()
        if not texto:
            continue

        analysis = sentiment_model(texto, truncation=True)
        pred = analysis[0]
        label = pred.get("label", "").lower()
        score = round(pred.get("score", 0.0), 3)

        if label == "positive":
            sentimento = "Positivo"
        elif label == "negative":
            sentimento = "Negativo"
        else:
            sentimento = "Neutro"

        resultados.append({
            "id":         c.get("id"),
            "id_autor":   c.get("id_autor"),
            "comentario": texto,
            "data_criacao": c.get("data_criacao"),
            "score":      score,
            "sentimento": sentimento
        })

    if not resultados:
        raise HTTPException(
            status_code=404,
            detail=f"Nenhum comentário para id {id_proposta}"
        )

    return {
        "id_proposta": id_proposta,
        "quantidade":  len(resultados),
        "comentarios": resultados
    }
