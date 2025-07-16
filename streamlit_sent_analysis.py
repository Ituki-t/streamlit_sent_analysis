import streamlit as st
import MeCab as mc
import pandas as pd
import numpy as np
import os
from pprint import pprint

st.title("感情分析")
sent = st.text_input("文章をを入力してください")


def hiragana2katakana(text):
    """ひらがなをカタカナへ変換
    """
    
    # カタカナの Unicode範囲 0x30A1 (ァ) から 0x30F6 (ヶ)
    # ひらがなの Unicode範囲 0x3041 (ぁ) から 0x3096 (ゖ)
    katakana_start = ord("ア")
    hiragana_start = ord("あ")

    #ひらがな2カタカタの辞書
    hiragana2katakana_map = {
        chr(i): chr(i - hiragana_start + katakana_start)for i in range(hiragana_start, ord("ゖ") + 1)
    }

    return text.translate(str.maketrans(hiragana2katakana_map))


def get_lemma_score(dic_lemma, lemma):
    """
    lemma(原型)で検索して感情スコアを返す。
    resultsが1個の場合スコアを返し、また、resultsが1より大きい場合、原型で返す 。
    その他はnan
    この段階では{reading:score}しか返していない
    """
    results = dic_lemma.get(lemma, {})
    if len(results) == 1:
        return next(iter(results.values())) # valueはreading と score ?
    elif len(results) > 1:
        score = results.get(lemma, np.nan) # get(key)はvalueを返す
        return score 
    return np.nan


def get_reading_score(dic_reading, reading, lemma):
    """
    読みで検索してスコアを返す
    ただし、読みが同じで、原型については辞書に登録された原型の文字数分が同じであるものだけ
    つまり、「ホームラン」の「ホーム」が一致する場合はbスコアを返すが、「タ」や「デ」についてはスコアw返さない
    """
    results = dic_reading.get(reading, {})
    if len(results) == 1:
        key, score = next(iter(results.items()))
        if key == lemma[:len(key)]:
            return score
    return np.nan


def get_sentiment_score(dic_lemma, dic_reading, text):
    """
    形態素解析をした後に、単語感情極性実数地を取得し、辞書リスト形式で返す
    """
    t = mc.Tagger()
    node = t.parseToNode(text)
    words = []
    while(node):
        if node.surface != "":
            pos = node.feature.split(",")[0]
            lemma = node.feature.split(",")[6]
            reading = node.feature.split(",")[7]
            score = get_lemma_score(dic_lemma, lemma)
            # この段階でget_lemma_score()で作った{reading:score}辞書から適切なreadingを選ぶ
            if np.isnan(score):
                score = get_reading_score(dic_reading, reading, lemma)
            w = {
                'surface': node.surface,
                'lemma': lemma,
                'reading': reading,
                'pos': pos,
                'score': score
            }
            words.append(w)
        node = node.next
        if node is None:
            break
    return words


# 単語感情性対応表
path_dic = os.path.sep.join(['dic', 'pn_ja.dic'])
df_pn = pd.read_csv(path_dic, encoding='sjis', sep=':', names=["lemma", "reading", "pos", "score"])


# 読みをすべてカタカナへ変換
df_pn['reading'] = [hiragana2katakana(v) for v in df_pn['reading'].tolist()]


# データフレームを辞書に変換
dic_lemma = df_pn.groupby('lemma').apply(lambda x: dict(zip(x['reading'], x['score']))).to_dict()
dic_reading = df_pn.groupby('reading').apply(lambda x: dict(zip(x['lemma'], x['score']))).to_dict()

# sent = st.text_input("文章をを入力してください")

results = get_sentiment_score(dic_lemma, dic_reading, sent)
df = pd.DataFrame(results)
st.write(f"{df['score'].mean():7.4f} : {sent}")
st.write(df)