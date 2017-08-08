# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import unicode_literals

text = '''
俗稱「菜瓜布肺」的特發性肺纖維化，有人形容「奪命狠過癌症」，醫師指出，以往國人對菜瓜布肺很陌生，導致延誤診斷，現透過胸部Ｘ光可初步篩檢，有機會抓出潛在患者。提醒大家，若乾咳逾兩個月，日常活動愈來愈喘，應盡速就醫檢查。

全台菜瓜布肺患者約一千三百名，陽明大學附設醫院胸腔內科主任張時杰表示，菜瓜布肺是漸進性纖維化間質性肺炎。

張時杰說，以往治療僅能以類固醇、免疫抑制劑緩和症狀，患者肺功能銳減速度為常人六倍，肺活量每年以下降兩百毫升的速度惡化，一旦患者急性發作、喘不過氣，可能因搶救不及而心肺衰竭。

目前健保已通過治療藥物，可減緩肺功能下降近五成，急性惡化機率減少六成八，如持續治療，能降低死亡風險四成三；已有患者用藥後惡化情形趨緩，從無法活動到每日步行一點五公里，提高生活品質，引起全台胸腔內科醫師大規模搜查疑似病例，欲揪出潛在患者。

張時杰表示，初步篩檢可透過公司企業或自費進行胸部Ｘ光合併吹氣肺功能檢查，後續再以臨床診斷、高解析度電腦斷層確診。

菜瓜布肺的危險因子，包含逾五十歲、直系家人或近親曾有病史、吸菸、暴露於化學工廠環境作業員等風險較高。
'''

from snownlp import normal
from snownlp import seg
from snownlp import sentiment
from snownlp.summary import textrank

if __name__ == '__main__':
    
    sents = normal.get_sentences(text)
    doc = []
    
    #summary 
    for sent in sents:
        words = seg.seg(sent)
        words = normal.filter_stop(words)
        doc.append(words)
    rank = textrank.TextRank(doc)
    rank.solve()
    for index in rank.top_index(5):
        print(sents[index])

    #probability of the sentiment 
    pro=sentiment.classify(text)
    print(pro)
