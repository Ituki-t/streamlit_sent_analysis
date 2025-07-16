import streamlit as st
from janome.tokenizer import Tokenizer
import pandas as pd
import numpy as np
import os

st.title("感情分析（Janome版）")
sent = st.text_input("文章を入力してください")


def hiragana2katakana(text):
    """ひらがなをカタカナへ変換"""
    katakana_start = ord("ア")
    hiragana_start = ord("あ")
    hiragana2katakana_map = {
        chr(i): chr(i - hiragana_start + katakana_start)
        for i in range(hiragana_start, ord("ゖ") + 1)
    }
    return text.translate(str.maketrans(hiragana2katakana_map))


def get_lemma_score(dic_lemma, lemma):
    """lemma(原型)で検索して感情スコアを返す"""
    results = dic_lemma.get(lemma, {})
    if len(results) == 1:
        return next(iter(results.values()))
    elif len(results) > 1:
        score = results.get(lemma, np.nan)
        return score
    return np.nan


def get_reading_score(dic_reading, reading, lemma):
    """読みで検索してスコアを返す"""
    results = dic_reading.get(reading, {})
    if len(results) == 1:
        key, score = next(iter(results.items()))
        if key == lemma[:len(key)]:
            return score
    return np.nan


def get_sentiment_score(dic_lemma, dic_reading, text):
    """Janomeで形態素解析 → 感情スコアを計算"""
    tokenizer = Tokenizer()
    words = []
    for token in tokenizer.tokenize(text):
        surface = token.surface
        pos = token.part_of_speech.split(',')[0]
        lemma = token.base_form if token.base_form != "*" else surface
        reading = token.reading if token.reading != "*" else surface
        reading = hiragana2katakana(reading)

        score = get_lemma_score(dic_lemma, lemma)
        if np.isnan(score):
            score = get_reading_score(dic_reading, reading, lemma)

        words.append({
            'surface': surface,
            'lemma': lemma,
            'reading': reading,
            'pos': pos,
            'score': score
        })
    return words


# 単語感情性対応表
path_dic = os.path.sep.join(['dic', 'pn_ja.dic'])
df_pn = pd.read_csv(path_dic, encoding='sjis', sep=':', names=["lemma", "reading", "pos", "score"])

# 読みをすべてカタカナへ変換
df_pn['reading'] = [hiragana2katakana(v) for v in df_pn['reading'].tolist()]

# データフレームを辞書に変換
dic_lemma = df_pn.groupby('lemma').apply(lambda x: dict(zip(x['reading'], x['score']))).to_dict()
dic_reading = df_pn.groupby('reading').apply(lambda x: dict(zip(x['lemma'], x['score']))).to_dict()

# 感情スコア計算
if sent:
    results = get_sentiment_score(dic_lemma, dic_reading, sent)
    df = pd.DataFrame(results)
    st.write(f"平均スコア: {df['score'].mean():7.4f} | 入力文: {sent}")
    st.write(df)
