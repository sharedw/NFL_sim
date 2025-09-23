import pandas as pd
from numpy import nan

ref2pp_tm = {
	'ARI': 'AZ',
	'CHW': 'CWS',
	'KCR': 'KC',
	'SDP': 'SD',
	'SFG': 'SF',
	'TBR': 'TB',
	'WSN': 'WSH',
}


def process_batting(url):
	bats = pd.read_csv(url, index_col=0)
	bat2pp = {
		'R': 'Runs',
		'H': 'Hits',
		'2B': 'Doubles',
		'3B': 'Triples',
		'HR': 'Home Runs',
		'Tm': 'team',
		'Opp': 'opp',
		'SB': 'Stolen Bases',
		'Date': 'date',
		'RBI': 'RBIs',
		'DFS(DK)': 'Hitter Fantasy Score',
		'SO': 'Hitter Strikeouts',
		'BB': 'Walks',
	}
	cols = bats.columns
	bats.columns = [bat2pp.get(x, x) for x in cols]
	bats.rename({'Unnamed: 6': 'home'}, inplace=True, axis=1)
	bats['home'] = bats['home'].replace({'@':0}).fillna(1)
	bats['team'] = bats['team'].apply(
		lambda x: ref2pp_tm.get(
			x,
			x,
		)
	)
	bats['opp'] = bats['opp'].apply(
		lambda x: ref2pp_tm.get(
			x,
			x,
		)
	)
	bats['park'] = bats.apply(lambda x: x['team'] if x['home'] == 1 else x['opp'], axis=1)
	bats['date'] = bats['date'].apply(
		lambda x: x.split('(')[0].split('\xa0susp')[0] + ' 2024'
	)
	bats['date'] = pd.to_datetime(bats['date'])
	bats['Hits+Runs+RBIs'] = bats['Hits'] + bats['RBIs'] + bats['Runs']
	bats['Singles'] = (
		bats['Hits'] - bats['Doubles'] - bats['Triples'] - bats['Home Runs']
	)
	bats['Total Bases'] = bats['Singles'] + (
		2 * bats['Doubles'] + (3 * bats['Triples']) + (4 * bats['Home Runs'])
	)
	return bats


def process_pitching(url):
	pitchers = pd.read_csv(url, index_col=0)

	def get_outs(s):
		vals = str(s).split('.')
		return int(vals[0]) * 3 + int(vals[1])

	pitch2pp = {
		'R': 'Runs Allowed',
		'H': 'Hits Allowed',
		'2B': 'Doubles Allowed',
		'3B': 'Triples Allowed',
		'HR': 'Home Runs Allowed',
		'Tm': 'team',
		'Opp': 'opp',
		'SB': 'Stolen Bases',
		'Date': 'date',
		'RBI': 'RBIs',
		'DFS(DK)': 'Pitcher Fantasy Score',
		'SO': 'Pitcher Strikeouts',
		'BB': 'Walks Allowed',
		'ER': 'Earned Runs Allowed',
	}

	cols = pitchers.columns
	pitchers.columns = [pitch2pp.get(x, x) for x in cols]
	pitchers.rename({'Unnamed: 6': 'home'}, inplace=True, axis=1)
	pitchers['home'] = pitchers['home'].replace({'@':0}).fillna(1)

	pitchers['team'] = pitchers['team'].apply(
		lambda x: ref2pp_tm.get(
			x,
			x,
		)
	)
	pitchers['opp'] = pitchers['opp'].apply(
		lambda x: ref2pp_tm.get(
			x,
			x,
		)
	)
	pitchers['park'] = pitchers.apply(lambda x: x['team'] if x['home'] == 1 else x['opp'], axis=1)
	pitchers['date'] = pitchers['date'].apply(
		lambda x: x.split('(')[0].split('\xa0susp')[0] + ' 2024'
	)
	pitchers['date'] = pd.to_datetime(pitchers['date'])
	pitchers['Pitching Outs'] = pitchers['IP'].apply(lambda x: get_outs(x))
	pitchers['Quality Start'] = 0
	pitchers.loc[
		(pitchers['Pitching Outs'] >= 18) & (pitchers['Earned Runs Allowed'] <= 3),
		'Quality Start',
	] = 1
	w_l = {'W': 1, 'L': 0, 'nan': 0, 'H': 0, 'S': 0, 'BS': 0, 'BW': 0, 'BL': 0, 'HL': 0}
	pitchers['W/L'] = pitchers['Dec'].apply(lambda x: w_l[(str(x).split('(')[0])])
	pitchers['Pitcher Fantasy Score'] = (
		(6 * pitchers['W/L'])
		+ (4 * pitchers['Quality Start'])
		+ (3 * pitchers['Pitcher Strikeouts'])
		+ (pitchers['Pitching Outs'])
		- (3 * pitchers['Earned Runs Allowed'])
	)
	return pitchers


def get_favored(row):
	if row['elo_diff'] >= 45:
		return 1
	if row['elo_diff'] < 45 and row['elo_diff'] >= 0:
		return 0
	if row['elo_diff'] < 0 and row['elo_diff'] >= -40:
		return 0
	if row['elo_diff'] <= -40:
		return -1
	return nan


def pad_results(row):
	if row['alt_line'] == 'demon':
		if row['o_u'] == 'over':
			base = 6
		else:
			base = 14
	elif row['alt_line'] == 'goblin':
		if row['o_u'] == 'over':
			base = 13
		else:
			base = 7
	elif row['alt_line'] == 'standard':
		base = 10
	denom = base + (row['proportion'] * row['count'])
	numerat = row['count'] + 20
	return denom / numerat



def team_quality(elo):
	cutoff = 1335
	if elo >= cutoff:
		return 1
	elif elo < cutoff and elo > 1270:
		return 0 
	
	else: return -1
	return nan

def lineup_order(bop):
	cutoff = 4
	if bop >= cutoff:
		return 0
	elif bop < cutoff:
		return 1 
	return -1