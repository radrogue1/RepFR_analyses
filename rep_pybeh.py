import pandas as pd
import numpy as np

def abs_min_split(items):
    mini = []
    mini.append(items[0])
    for item in items[1:]:
        if abs(item) < abs(mini[0]):
            mini = []
            mini.append(item)
        elif abs(item) == abs(mini[0]):
            mini.append(item)
    mini = set(mini)
    mini = list(mini)
    return mini

def abs_min_pos(items):
    mini = []
    mini.append(items.pop(0))
    for item in items:
        if abs(item) < abs(mini[0]):
            mini = []
            mini.append(item)
        elif abs(item) == abs(mini[0]):
            mini.append(item)
    if len(mini) > 1:
        mini = abs(mini[0])
    return mini

def min_pos_lags(df):
    items = df['pos_lags'].to_list()
    mini = []
    mini.append(items.pop(0))
    for item in items:
        if abs(item) < abs(mini[0]):
            mini = []
            mini.append(item)
        elif abs(item) == abs(mini[0]):
            mini.append(item)
    if len(mini) > 1:
        mini = [abs(mini[0])]
    mini = mini[0]
    return mini

def get_act_spos(df, list_type = 'list', item_num = 'item_num'):
    recs = []
    rec_evs = df[df.type == 'REC_WORD']
    word_evs = df[df.type == 'WORD']
    for l, df in rec_evs.groupby(list_type):
        rec_words = df[item_num].unique()
        word_type = pd.DataFrame()
        word_type = word_evs[word_evs[list_type] == l]
        word_type['serialpos'] = word_type['serialpos'].astype(int)
        word_type = word_type[word_type[item_num].isin(rec_words)]
        words = word_type[item_num].unique()
        pos = pd.DataFrame(columns = [item_num, 'serialpos'])
        pos[item_num] = pd.Series(words)
        pos = pos.set_index(item_num)
        for word in words:
            pos.at[word, 'serialpos'] = word_type[word_type[item_num] == word].serialpos.unique()
        pos = pos.reset_index()
        pos = pos[pos[item_num].isin(df[item_num])]
        pos = pos.set_index(item_num)
        df = df.reset_index()
        df = df.set_index(item_num)
        df['act_serialpos'] = pos['serialpos']
        df.act_serialpos = df.act_serialpos.fillna(-1)
        df.reset_index(inplace = True)
        df.set_index('index', inplace = True)
        recs.append(df)
    recs = pd.concat(recs)
    evs = word_evs.append(recs)
    return evs



def crp(evs, num_lags, list_length, list_type = 'list', item_num = 'item_num'): 
    crps = []
    act_lags = np.zeros((2*list_length-1))
    pos_lags = np.zeros((2*list_length-1))
    recs = evs[evs.type == 'REC_WORD']
    for l, df in recs.groupby(list_type):
        used_positions = np.zeros(list_length)
        serialpos = df.act_serialpos.to_numpy()
        crp = pd.DataFrame(columns=['lag', 'prob'])
        crp['lag'] = pd.Series(range(-list_length +1, list_length))
        for i in range(len(serialpos)-1):
            try:
                for j, previous in enumerate(serialpos[i]):
                    used_positions[previous] +=1
                    for k, current in enumerate(serialpos[i+1]):
                        lag = current - previous
                        act_lags[lag+list_length-1] +=1

                open_pos = np.where(used_positions==0)
                pos_lag = []
                for j, position in enumerate(serialpos[i]):
                    pos_lag = open_pos - position
                    for k in pos_lag:
                        for p in k:
                            pos_lags[p+list_length-1] +=1
            except Exception as e:
                continue
    act_lags = np.array(act_lags)
    pos_lags = np.array(pos_lags)
    crp['prob'] = np.divide(act_lags, pos_lags)
    crp = crp[crp['lag'] >= -num_lags]
    crp = crp[crp['lag'] <= num_lags]

    return crp
   
def min_crp(evs, num_lags, list_length, halfornah ='nah', list_type = 'list', item_num = 'item_num'): 
    pos_lags = np.zeros((2*list_length-1))
    act_lags = np.zeros((2*list_length-1))
    for l, df in evs.groupby(list_type):
        rec_df = df[df.type == 'REC_WORD']
        enc_df = df[df.type == 'WORD']
        item_num_enc = enc_df[item_num].to_numpy()
        item_num_rec = rec_df[item_num].to_numpy()
        used_positions = np.zeros(list_length)
        serialpos = rec_df.act_serialpos.to_numpy()
        
        crp = pd.DataFrame(columns=['lag', 'prob'])
        crp['lag'] = pd.Series(range(-list_length +1, list_length))
        for i in range(len(serialpos)-1):
            temp_lags = []
            try:
                for j, previous in enumerate(serialpos[i]):
                    used_positions[previous] +=1
                    for k, current in enumerate(serialpos[i+1]):
                        if current - previous != 0:
                            temp_lags.append(current - previous) 
    #             MAKE SURE ALL OF THIS IS OUTSIDE OF FOR LOOPS
                if halfornah == 'nah':
                    lag = abs_min_pos(temp_lags)[0]
                    act_lags[lag+list_length-1]+=1
    #     Use this for half and half crp
                else:
                    lags = abs_min_split(temp_lags)
                    if len(lags) > 1:           
                        for lag in lags:
                            act_lags[lag+list_length-1] += 0.5
                    else:
                        act_lags[lags[0]+list_length-1]+=1
                open_pos, = np.where(used_positions==0)
                item_num_enc = enc_df[item_num].to_numpy()
                item_num_rec = rec_df[item_num].to_numpy()
                all_pos_lags = pd.DataFrame(pd.Series(np.arange(-list_length+1, list_length), name = 'pos_lags'))
                all_pos_lags.set_index('pos_lags', inplace = True)
                all_pos_lags[item_num] = pd.Series(np.nan, index = all_pos_lags.index)
                all_pos_lags.drop(0, 0, inplace = True)
                for spos, pres in enumerate(item_num_enc):
                    if spos in open_pos:
                        all_pos_lag = spos - serialpos[i]
                        all_pos_lags.at[all_pos_lag] = pres
                all_pos_lags.dropna(inplace=True)
                all_pos_lags = all_pos_lags.reset_index().groupby(item_num).apply(lambda x: min_pos_lags(x)) + list_length -1
                pos_lags[all_pos_lags] +=1
            except Exception as e:
                continue
    crp['prob'] = np.divide(act_lags, pos_lags)
    crp = crp[crp['lag'] >= -num_lags]
    crp = crp[crp['lag'] <= num_lags]

    return crp

def temp_percentile_rank(actual, possible):
    """
    Helper function to return the percentile rank of the actual transition within the list of possible transitions.

    :param actual: The distance of the actual transition that was made.
    :param possible: The list of all possible transition distances that could have been made.

    :return: The proportion of possible transitions that were more distant than the actual transition.
    """
    # If there were fewer than 2 possible transitions, we can't compute a meaningful percentile rank
    if len(possible) < 2:
        return None

    # Sort possible transitions from largest to smallest
    possible = sorted(possible)[::-1]

    # Get indices of the one or more possible transitions with the same distance as the actual transition
    matches, = np.where(possible == actual)

    if len(matches) > 0:
        # Get the number of possible transitions that were more distant than the actual transition
        # If there were multiple transitions with the same distance as the actual one, average across their ranks
        rank = np.mean(matches)
        # Convert rank to the proportion of possible transitions that were more distant than the actual transition
        ptile_rank = rank / (len(possible) - 1.)
    else:
        ptile_rank = None

    return ptile_rank

def temp_fact(evs, list_length): 
    recs = evs[evs.type == 'REC_WORD']
    total = 0
    count = 0
    for l, df in recs.groupby('list'):
        used_positions = np.zeros(list_length)
        serialpos = df.act_serialpos.to_numpy()
        crp = pd.DataFrame(columns=['lag', 'prob'])
        crp['lag'] = pd.Series(range(-list_length +1, list_length))
        for i in range(len(serialpos)-1):
            actual = []
            try:
                for j, previous in enumerate(serialpos[i]):
                    used_positions[previous] +=1
                    for k, current in enumerate(serialpos[i+1]):
                        lag = abs(current - previous)
                        actual.append(lag)
                open_pos = np.where(used_positions==0)
                possible = []
                for j, position in enumerate(serialpos[i]):
                    possible = abs(open_pos - position)[0]
                for lag in actual:
                    rank = temp_percentile_rank(lag, possible)
                    if rank is not None:
                        total += rank
                        count += 1
            except Exception as e:
                pass

    if count == 0:
                count= np.nan
    temp_fact = total / count
    return temp_fact



