import data_diversification

dd = data_diversification.DataDiversification(k=2, N=2, n_epoch=2)
dd.diversify(
    arch_fwd='transformer_iwslt_de_en', 
    arch_bkwd='transformer_wmt_en_de',
    src_lang='de', trg_lang='en')
'''
dd = data_diversification.DataDiversification(k=2, N=2, n_epoch=2)
dd.diversify(
    arch_fwd='transformer_iwslt_de_en', 
    arch_bkwd='transformer_wmt_en_de',
    src_lang='ne', trg_lang='en')
'''
