import random
import tls_client
from bs4 import BeautifulSoup
import pandas as pd
import os
import uuid
import time
from scipy.stats import poisson
import sqlite3 as sq


def load_current_line(path, time_col, identifier) -> pd.DataFrame:
	df = pd.read_parquet(path)
	if (time_col in df.columns) and (identifier in df.columns):
		df = df.sort_values(by=time_col, ascending=False)
		return df.loc[df[time_col] == df[time_col].max()].reset_index(drop=True)
	else:
		return df


def load_unabated_current(path, time_col, identifier) -> pd.DataFrame:
	df = pd.read_parquet(path)
	if (time_col in df.columns) and (identifier in df.columns):
		df = get_line_movement(df)
		df = df.sort_values(by=time_col, ascending=False)
		return df.loc[df[time_col] == df[time_col].max()].reset_index(drop=True)
	else:
		return df


def get_line_movement(df):
	diffs = df.groupby(['prop_id', 'scrape_time']).first().reset_index()
	diffs['last_pred'] = diffs.groupby(['prop_id'])['pred'].shift(1)
	diffs
	df = df.merge(
		diffs[['prop_id', 'scrape_time', 'last_pred']], on=['prop_id', 'scrape_time']
	)
	df['line_movement'] = 100 * (df['pred'] - df['last_pred']) / df['pred']
	df.loc[(df.prop_id == '30e8404f-c')]
	return df


def get_nba_results(lines, games):
	def calc_outcome(row):
		outcome = 0
		for part in row['stat'].split('+'):
			try:
				outcome += row[part]
			except KeyError:
				return float('nan')
		return outcome

	comb = lines.merge(games, on=['player', 'date', 'opp'])
	comb['result'] = comb.apply(lambda row: calc_outcome(row), axis=1)
	comb['o_u'] = [
		'under' if x > y else 'over' if x < y else 'push'
		for x, y in zip(comb['line'], comb['result'])
	]
	return comb


def create_prop_id(row):
	seed = (
		f"{row['player']}_{row['stat']}_{pd.to_datetime(row['event_time']).floor('h')}"
	)
	id = uuid.uuid5(uuid.NAMESPACE_DNS, seed)
	return str(id)[0:10]


def create_event_id(row):
	teams = ''.join(sorted([str(row['opp']), str(row['team'])]))
	seed = f"{teams}_{pd.to_datetime(row['event_time']).floor('h')}"
	event_id = uuid.uuid5(uuid.NAMESPACE_DNS, seed)
	return str(event_id)[0:10]


def update_csv_file(new_file, file_path):
	if os.path.exists(file_path) is False:
		new_file.to_csv(file_path)
		print(f"{file_path.split('/')[-1]} created")
	else:
		old_file = pd.read_csv(file_path, index_col=0)
		pd.concat((new_file, old_file)).to_csv(file_path)
		print(f"{file_path.split('/')[-1]} updated")
	return


def update_parquet_file(new_file, file_path):
	if os.path.exists(file_path) is False:
		new_file.to_parquet(file_path)
		print(f"{file_path.split('/')[-1]} created")
	else:
		old_file = pd.read_parquet(file_path)
		pd.concat((new_file, old_file)).to_parquet(file_path)
		print(f"{file_path.split('/')[-1]} updated")
	return


def tls_get(url, headers=None):
	client_list = [
		'chrome_103',
		'chrome_105',
		'chrome_108',
		'chrome_110',
		'chrome_112',
		'firefox_102',
		'firefox_104',
		'opera_90',
		'safari_15_3',
		'safari_16_0',
		'safari_ios_15_5',
		'safari_ios_16_0',
		'okhttp4_android_12',
		'okhttp4_android_13',
	]
	requests = tls_client.Session(
		client_identifier=random.choice(client_list), random_tls_extension_order=True
	)
	try:
		return requests.get(url, headers=headers)
	except tls_client.exceptions.TLSClientExeption:
		return requests.get(url, headers=headers)


def get_url_soup(url, headers=None):
	try:
		response = tls_get(url, headers=headers)
	except tls_client.exceptions.TLSClientExeption:
		time.sleep(2)
		response = tls_get(url, headers=headers)
	return BeautifulSoup(response.content, features='lxml')


def get_url_json(url, headers=None):
	response = tls_get(url, headers=headers)
	return response.json()


def get_all_game_logs(parse_date=False):
	conn = sq.connect('C:/Github/NBA_betting_model/databases/games.sqlite')
	df = pd.read_sql_query('select * from games', conn)
	conn.close()
	if parse_date:
		df['date'] = (
			pd.to_datetime(df['date'], utc=True)
			.dt.tz_convert('US/Central')
			.dt.floor('d')
		)
	return df


def append_table_no_dupes(database_loc: str, table: str, new_data, primary_keys: list):
	conn = sq.connect(database_loc, autocommit=True)
	existing_rows = pd.read_sql_query(f'select * from {table}', conn)
	temp = existing_rows[primary_keys].merge(new_data, how='outer', indicator=True)
	new_data = (
		temp.loc[temp._merge == 'right_only']
		.drop('_merge', axis=1)
		.reset_index(drop=True)
	)
	new_data.to_sql(table, conn, if_exists='append', index=False)
	conn.commit()
	conn.close()
	return


def get_all_pp(closing_line=True):
	conn = sq.connect('C:/Github/NBA_betting_model/databases/lines.sqlite')
	pp = pd.read_sql_query('select * from lines', conn)
	conn.close()
	pp = pp.sort_values(by='date')
	if closing_line:
		pp = pp.groupby(['date', 'player', 'stat']).last().reset_index()
	pp['date'] = (
		pd.to_datetime(pp['date'], format='mixed', yearfirst=True, utc=True)
		.dt.tz_localize('US/Central')
		.dt.floor('d')
	)
	return pp


def american2prob(odds):
	prob = (-odds / (100 - odds)) if odds < 0 else 100 / (odds + 100)
	return prob


def mean2prob(mean, line, side):
	over = 1 - poisson.cdf(line, mean)
	under = poisson.cdf(line, mean) - poisson.pmf(line, mean)
	if side == 'over':
		return over / (over + under)
	if side == 'under':
		return under / (over + under)


def prob2american(p):
	if p > 0.5:
		# p = 100*p
		amer = -100 / (p - 1)
		amer = -amer + 100
	elif p < 0.5:
		amer = 100 * (1 / p)
		amer = amer - 100
	else:
		amer = 100
	return amer


def prob2mean(prob, line, side):
	# poisson is criteria then avg
	avg = line
	over = 1 - poisson.cdf(line, avg)
	under = poisson.cdf(line, avg) - poisson.pmf(line, avg)
	pois = over / (under + over)

	while abs(pois - prob) > 0.001:
		lr = avg * 0.5
		over = 1 - poisson.cdf(line, avg)
		under = poisson.cdf(line, avg) - poisson.pmf(line, avg)
		if side == 'over':
			pois = over / (under + over)
			if pois < prob:
				avg = (lr * (prob - pois)) + avg
			else:
				avg = avg - (pois - prob)
		else:
			pois = under / (under + over)
			if pois > prob:
				avg = (-lr * (prob - pois)) + avg
			else:
				avg = avg + lr * (pois - prob)
	return avg


def to_notz_date(date: pd.Series):
	cst_date = (
		pd.to_datetime(date, format='mixed', yearfirst=True)
		.dt.tz_localize(None)
		.dt.floor('d')
	)
	return cst_date


def check_line(df):
	def calc_outcome(row):
		outcome = 0
		for part in row['stat'].split('+'):
			try:
				outcome += row[part]
			except KeyError:
				return float('nan')
		return outcome

	df['result'] = df.apply(lambda row: calc_outcome(row), axis=1)
	df['o_u'] = [
		'under' if x > y else 'over' if x < y else 'push'
		for x, y in zip(df['line'], df['result'])
	]
	return df


def score_prediction(home_rating, away_rating):
	return (home_rating - away_rating) / 28.0


# get ft closing line df
def get_stat_close(stat):
	with open('C:/Github/NBA_betting_model/databases/stat_close.sql', 'r') as query:
		conn = sq.connect('databases/lines.sqlite')
		pp = pd.read_sql_query(query.read(), conn, params=[stat])
		conn.close()
	pp = pp.sort_values(by='date')
	return pp


def pivot_lines(df, col):
	x = (
		df.groupby(col)['o_u']  # .loc[df.stat == stat]
		.value_counts()
		.reset_index()
		.copy(deep=True)
	)
	base_rate = {'demon': 2.5, 'standard': 5, 'goblin': 6.5}
	pivot_df = x.pivot(index=col, columns='o_u', values='count').fillna(0).reset_index()
	if 'push' in pivot_df.columns:
		pivot_df['push'] = pivot_df['push'].fillna(0)
	else:
		pivot_df['push'] = 0

	if 'over' in pivot_df.columns:
		pivot_df['over'] = pivot_df['over'].fillna(0)
	else:
		pivot_df['over'] = 0

	if 'under' in pivot_df.columns:
		pivot_df['under'] = pivot_df['under'].fillna(0)
	else:
		pivot_df['under'] = 0
	pivot_df['over_ratio'] = (
		pivot_df['over'] + pivot_df['alt_line'].apply(lambda x: base_rate[x])
	) / (10 + pivot_df['under'] + pivot_df['over'])
	pivot_df['count'] = pivot_df['over'] + pivot_df['under'] + pivot_df['push']

	return pivot_df


def get_results(lines, games):
	def calc_outcome(row):
		outcome = 0
		for part in row['stat'].split('+afsda'):
			try:
				outcome += row[part]
			except KeyError:
				return float('nan')
		return outcome

	comb = lines.merge(games, on=['player', 'date', 'team', 'opp'])
	comb['result'] = comb.apply(lambda row: calc_outcome(row), axis=1)
	comb['o_u'] = [
		'under' if x > y else 'over' if x < y else 'push' if x == y else 'N/A'
		for x, y in zip(comb['line'], comb['result'])
	]
	return comb.loc[comb.o_u != 'N/A'].reset_index(drop=True)


def value_count_plus(df, grouper, col):
	normalized = df.groupby(grouper)[col].value_counts(normalize=True).reset_index()
	raw = df.groupby(grouper)[col].value_counts().reset_index()
	return normalized.merge(raw).sort_values(by=grouper + col)


def get_pp_close():
	pp = pd.read_parquet('C:/Github/NBA_betting_model/reference_data/pp_close.parquet')
	pp['alt_line'] = pp['alt_line'].fillna('standard')
	pp['date'] = pd.to_datetime(pp['date'], utc=True).dt.tz_convert(None).dt.round('d')
	return pp


def get_corr(one, two,three=pd.DataFrame(), grouper=[], relation='same'):
    df1 = one.copy(deep=True)
    df2 = two.copy(deep=True)
    df3 = three.copy(deep=True)
    #get cols to merge on
    grouper = ['alt_line','stat'] + grouper
    home_cols = ['team', 'date']
    opp_cols = ['opp', 'date']
    if relation=='same':
        right_cols = home_cols
    else:
        right_cols = opp_cols

    #fixing weird column renaming
    if not df3.empty:
        a = 4
    else:
        a = 3
    count_cols = []
    o_u_cols = []
    for x in grouper:
        for n in range(1,a):
            count_cols += [x+f'_{n}']
            o_u_cols += [f'o_u_{n}']
    o_u_cols = sorted(list(set(o_u_cols)))

    raw_cols = list(set(grouper + home_cols + right_cols + ['o_u'] + ['player']))
    
    #merge dfs
    comb_corr = df1.loc[(df1.alt_line!='goblin') & (df1.o_u != 'push'), raw_cols].merge(
        df2.loc[(df2.alt_line!='goblin') & (df2.o_u != 'push'), raw_cols],
        left_on=home_cols,
        right_on=right_cols,
        suffixes=('_1', '_2'),
        )
    #filter out same player corr and mirrored lines
    comb_corr = comb_corr.loc[(comb_corr.player_1 != comb_corr.player_2) & 
                              ~((comb_corr.player_1 < comb_corr.player_2) & (comb_corr.stat_1 == comb_corr.stat_2))
                              ].reset_index(drop=True)
    if not df3.empty:
        comb_corr = comb_corr.merge(
            df3.loc[(df3.alt_line!='goblin'), raw_cols],
            left_on=home_cols,
            right_on=right_cols,
            #suffixes=('', '_2'),
            )
        comb_corr.columns = [x+'_3' if (x in raw_cols) else x for x in comb_corr.columns]
        comb_corr = comb_corr.loc[comb_corr.player_1 != comb_corr.player_3].reset_index(drop=True)
        comb_corr = comb_corr.loc[comb_corr.player_2 != comb_corr.player_3].reset_index(drop=True)
    corr = value_count_plus(
        comb_corr,
        count_cols,
        o_u_cols,
    )
    corr['relation'] = relation
    corr = corr.loc[~((corr.alt_line_1.isin(['demon', 'goblin'])) & (corr.o_u_1 == 'under'))]
    corr = corr.loc[~((corr.alt_line_2.isin(['demon', 'goblin'])) & (corr.o_u_2 == 'under'))]
    if not df3.empty:
        corr = corr.loc[~((corr.alt_line_3.isin(['demon', 'goblin'])) & (corr.o_u_3 == 'under'))]

    corr = corr.loc[corr['count'] > 0].sort_values(by='proportion', ascending=False)
    corr['equiv'] = corr['proportion'] ** (1/(a-1))
    corr = corr.rename({'count':'count_all'},axis=1)

    #filtering extraneous permutations
    numbered_cols = []
    for num in range(1,a):
        numbered_cols.append(num)
        corr[num] = ''
        for col in (['stat', 'alt_line', 'o_u'] + grouper):
            num_col = f'{col}_{num}'
            corr[num] += corr[num_col].astype(str)
    if a == 4:
        corr['concat'] = corr.apply(lambda x: str(sorted(list((x[1],x[2],x[3])))),axis=1)
    else:
        corr['concat'] = corr.apply(lambda x: str(sorted(list((x[1],x[2])))),axis=1)
    corr = corr.groupby(['concat']).head(1)
    corr = corr.sort_values(by='equiv',ascending=False).drop([1,2,3,'concat'],axis=1,errors='ignore')

    return corr.sort_values(by='proportion',ascending=False)

def weight_corr(corr, hit_df,suff):
    merge_cols = []
    for col in hit_df.columns:
        if 'hit_' in col:
            pass
        elif col != ('count') and col != ('proportion'):
            merge_cols.append(col)
    for n in range(1,4):
        n_merge_cols = [col + f'_{n}' for col in merge_cols]
        if f'alt_line_{n}' in corr.columns:
            corr = corr.merge(
                hit_df,
                left_on=n_merge_cols,
                right_on=merge_cols,
                how='left',
                suffixes=[f'_{n-1}',f'_{n}']
            )
            corr.drop(merge_cols, axis=1, inplace=True)
            corr.rename({f'hit_{suff}': f'hit_{suff}_{n}'},axis=1,inplace=True)
    corr['expected'] = 1
    for n in range(1,4):
        col = f'hit_{suff}_{n}'
        if col in corr.columns:
            corr['expected'] = corr['expected'] * corr[col]
    corr['diff'] = corr['proportion'] - corr['expected']
    return corr