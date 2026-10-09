import akshare as ak

# stock_board_industry_name_em_df = ak.stock_board_industry_name_em()
# print(stock_board_industry_name_em_df)

# stock_board_industry_spot_em_df = ak.stock_board_industry_spot_em(symbol="小金属")
# print(stock_board_industry_spot_em_df)

# stock_hsgt_board_rank_em_df = ak.stock_hsgt_board_rank_em(symbol="北向资金增持行业板块排行", indicator="今日")
# print(stock_hsgt_board_rank_em_df)

# stock_hsgt_hist_em_df = ak.stock_hsgt_hist_em(symbol="北向资金")
# print(stock_hsgt_hist_em_df)

# stock_board_concept_cons_em_df = ak.stock_board_concept_cons_em(symbol="融资融券")
# print(stock_board_concept_cons_em_df)

stock_board_concept_name_em_df = ak.stock_board_concept_name_em()
stock_board_concept_name_em_df.to_excel('概念板块列表.xlsx', index=False)
print(stock_board_concept_name_em_df)