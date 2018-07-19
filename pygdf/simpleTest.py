# Copyright (c) 2018, NVIDIA CORPORATION.

import numpy as np
import pandas as pd

from pygdf.dataframe import DataFrame


def make_frame(dataframe_class, nelem, seed=0, extra_levels=()):
    np.random.seed(seed)

    df = dataframe_class()

    df['x'] = np.random.randint(0, 5, nelem)
    df['y'] = np.random.randint(0, 3, nelem)
    for lvl in extra_levels:
        df[lvl] = np.random.randint(0, 2, nelem)

    df['val'] = np.random.random(nelem)

    return df



def test_groupby_mean(nelem = 3):
    # gdf
    
    df = make_frame(DataFrame, nelem=nelem)
    
#     from pygdf.groupby import Groupby
    from pygdf.libgdf_groupby import LibGdfGroupby
    
    by = ('x', 'y')
#     gb = Groupby(df, by=by)
    
    newgd = LibGdfGroupby(df, by=by)
    
    newgot_df = newgd.mean()
    
    print(newgot_df)
    print("done")
    
    pass
    
#     got_df = make_frame(DataFrame, nelem=nelem).groupby(('x', 'y')).mean()
#     got_df = gb.mean()
#     got = np.sort(got_df['val'].to_array())
#     # pandas
#     expect_df = make_frame(pd.DataFrame,
#                            nelem=nelem).groupby(('x', 'y')).mean()
#     expect = np.sort(expect_df['val'].values)
#     
#     print(got)
#     # verify
#     np.testing.assert_array_almost_equal(expect, got)


test_groupby_mean(3)


# 
# 
# 
# def test_dataframe_join_how():
#     
#     print("test_dataframe_join_how inside")
#     
#     np.random.seed(0)
# 
# #     hows = 'left,inner,outer,right'.split(',')
#     how = 'inner'
#     aa = [0, 0, 4, 5, 5]
#     bb = [0, 0, 2, 3, 5]
# 
# #     # Test specific cases (1)
# #     aa = [0, 0, 4, 5, 5]
# #     bb = [0, 0, 2, 3, 5]
# #     for how in hows:
# #         yield (aa, bb, how)
# # 
# #     # Test specific cases (2)
# #     aa = [0, 0, 1, 2, 3]
# #     bb = [0, 1, 2, 2, 3]
# #     for how in hows:
# #         yield (aa, bb, how)
# # 
# #     # Test large random integer inputs
# #     aa = np.random.randint(0, 50, 100)
# #     bb = np.random.randint(0, 50, 100)
# #     for how in hows:
# #         yield (aa, bb, how)
# # 
# #     # Test floating point inputs
# #     aa = np.random.random(50)
# #     bb = np.random.random(50)
# #     for how in hows:
# #         yield (aa, bb, how)
#     
#     
#     df = DataFrame()
#     df['a'] = aa
#     df['b'] = bb
# 
#     def work(df):
# #         ts = timer()
#         df1 = df.set_index('a')
#         df2 = df.set_index('b')
#         joined = df1.join(df2, how=how, sort=True)
# #         te = timer()
# #         print('timing', type(df), te - ts)
#         return joined
# 
#     expect = work(df.to_pandas())
#     got = work(df)
#     expecto = expect.copy()
#     goto = got.copy()
# 
#     # Type conversion to handle NoneType
#     expectb = expect.b
#     expecta = expect.a
#     gotb = got.b
#     gota = got.a
#     got.drop_column('b')
#     got.add_column('b', gotb.astype(np.float64).fillna(np.nan))
#     got.drop_column('a')
#     got.add_column('a', gota.astype(np.float64).fillna(np.nan))
#     expect.drop(['b'], axis=1)
#     expect['b'] = expectb.astype(np.float64).fillna(np.nan)
#     expect.drop(['a'], axis=1)
#     expect['a'] = expecta.astype(np.float64).fillna(np.nan)
# 
#     # print(expect)
#     # print(got.to_string(nrows=None))
# 
#     assert list(expect.columns) == list(got.columns)
#     assert np.all(expect.index.values == got.index.values)
#     if(how != 'outer'):
#         pd.util.testing.assert_frame_equal(
#             got.to_pandas().sort_values(['b', 'a']).reset_index(drop=True),
#             expect.sort_values(['b', 'a']).reset_index(drop=True))
#         # if(how=='right'):
#         #     _sorted_check_series(expect['a'], expect['b'],
#         #                          got['a'], got['b'])
#         # else:
#         #     _sorted_check_series(expect['b'], expect['a'], got['b'],
#         #                          got['a'])
#     else:
#         _check_series(expecto['b'], goto['b'])
#         _check_series(expecto['a'], goto['a'])
#         
#     return "done"
# 
# print("test_dataframe_join_how outside")        
#         
# test_dataframe_join_how()
# 
# print("test_dataframe_join_how end")
