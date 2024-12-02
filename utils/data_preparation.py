import pandas as pd

agg_funcs = ['mean', 'sum', 'min', 'max', 'count']

def aggregate_bureau_balance(bureau, bureau_balance, agg_funcs = agg_funcs):
    bureau_balance_agg = bureau_balance.groupby('SK_ID_BUREAU').agg(
        {col: agg_funcs for col in bureau_balance.columns if
         col != 'SK_ID_BUREAU'}
    )

    bureau_balance_agg.columns = [f"{col[0]}_{col[1]}" for col in
                                  bureau_balance_agg.columns]
    bureau_balance_agg.reset_index(inplace=True)

    bureau = bureau.merge(bureau_balance_agg, on='SK_ID_BUREAU', how='left')
    return bureau

def aggregate_table_by_prev(table, table_name, agg_funcs = agg_funcs):
    agg = table.groupby('SK_ID_PREV').agg(
        {col: agg_funcs for col in table.columns if col != 'SK_ID_PREV'}
    )
    agg.columns = [f"{table_name}_{col[0]}_{col[1]}" for col in
                    agg.columns]
    agg.reset_index(inplace=True)
    return agg

def aggregate_previous_application(previous_application, credit_card_balance,
                                   installments_payments, pos_cash_balance):

    credit_card_agg = aggregate_table_by_prev(
        credit_card_balance,'credit_card', agg_funcs = agg_funcs)
    installments_agg = aggregate_table_by_prev(
        installments_payments,'installments', agg_funcs = agg_funcs)
    pos_cash_agg = aggregate_table_by_prev(
        pos_cash_balance, 'pos_cash', agg_funcs = agg_funcs)

    previous_application = previous_application.merge(credit_card_agg,
                                                      on='SK_ID_PREV',
                                                      how='left')
    previous_application = previous_application.merge(installments_agg,
                                                      on='SK_ID_PREV',
                                                      how='left')
    previous_application = previous_application.merge(pos_cash_agg,
                                                      on='SK_ID_PREV',
                                                      how='left')

    return previous_application
