import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def pay_general(
	parlay_size: int, results: list, payout: list, normalize=False
) -> float:
	"""parlay_size is the number of legs per parlay
	results is a list of the number of legs that per parlay
	payout is the amount paid out from getting a certain number correct;
	payout[0] is result of hitting all legs, payout[1] missing one, etc.
	missing values at end of payout are assumed to be losing
	returns total profit unless normalize=True then returns profit per bet"""
	profit = 0
	for i in range(len(payout)):
		profit += results.count(parlay_size - i) * payout[i]
	profit -= len(results)
	if normalize:
		profit = profit / len(results)
	return profit


def prob(parlay_size: int, n: int, base_percent: float, other_legs=None, hit_12=None, hit_34=None):
	"""generates a list of parlay slip results
	parlay size: number of legs per parlay
	n: number of parlays
	percent: hit rate of the legs of each parlay
	"""

	if not other_legs:
		percent = [base_percent for _ in range(parlay_size)]
	else:
		percent =other_legs + [
			base_percent for _ in range(parlay_size - len(other_legs))
		]

	random_numbers = np.random.rand(n, parlay_size)

	if hit_12:
		corr_12 = np.random.rand(n)
		corr_12 = np.where(corr_12 > (1 - hit_12), 1, corr_12)  # both hit set to 1
		corr_12 = np.where(
			corr_12 < 1 - (2 * base_percent) + hit_12, 0, corr_12
		)  # neither hit set to 0
		corr_12 = np.tile(corr_12, (2, 1)).T
		corr_12 = np.where(corr_12 == 0, 1, np.where(corr_12 == 1, 0, 0.5))
		corr_12[(corr_12[:, 0] != 0) & (corr_12[:, 0] != 1), 1] = 1 #split the rest
		corr_12[(corr_12[:, 0] != 0) & (corr_12[:, 0] != 1), 0] = 0 #split the rest
		random_numbers[:, :2] = corr_12

	if hit_34:
		corr_34 = np.random.rand(n)
		corr_34 = np.where(corr_34 > (1 - hit_34), 1, corr_34)  # both hit set to 1
		corr_34 = np.where(
			corr_34 < 1 - (2 * base_percent) + hit_34, 0, corr_34
		)  # neither hit set to 0
		corr_34 = np.tile(corr_34, (2, 1)).T
		corr_34 = np.where(corr_34 == 0, 1, np.where(corr_34 == 1, 0, 0.5))
		corr_34[(corr_34[:, 0] != 0) & (corr_34[:, 0] != 1), 1] = 1 #split the rest
		corr_34[(corr_34[:, 0] != 0) & (corr_34[:, 0] != 1), 0] = 0 #split the rest
		random_numbers[:, :2] = corr_34


	successes = np.sum(random_numbers < percent, axis=1)
	random_numbers = np.round(random_numbers, 2)
	return successes.tolist()


def find_breakeven(
	num_demons=1,
	legs=5,
	payout=[10, 2, 0.4],
	base_hit_rate=0.53,
	hit_12=None,
	hit_34=None
):
	i = 0
	profit = 1
	demon_wr = 0.20
	if num_demons > 0:
		while (profit > 0.001 or profit < -0.001) and i < 1000:
			result = prob(
				legs, 100000, base_hit_rate, [demon_wr] * num_demons, hit_12=hit_12, hit_34=hit_34
			)
			profit = pay_general(legs, results=result, payout=payout, normalize=True)
			demon_wr -= 0.01 * profit * 5
			i += 1
		return demon_wr, base_hit_rate

	else:
		while (profit > 0.001 or profit < -0.001) and i < 1000:
			result = prob(legs, 100000, base_hit_rate, hit_12=hit_12, hit_34=hit_34)
			profit = pay_general(legs, results=result, payout=payout, normalize=True)
			base_hit_rate -= 0.01 * profit * 5
			i += 1
		return demon_wr, base_hit_rate


def betsim(
	bets,
	betsize_init,
	edge,
	bankroll,
	slip_size,
	other_legs=0.4,
	payout=[38, 4, 0.4],
	proportional_staking=False,
):
	slip_size = slip_size
	edge = edge + 0.01 * (np.random.random() - 0.5)
	x = []
	y = []

	slip_result = prob(slip_size, bets, edge, other_legs)
	num_wins = np.sum(np.array(slip_result) > 5)

	betsize = betsize_init

	for i in range(bets):
		if bankroll < 0:
			x.append(i)
			y.append(0)
		else:
			if proportional_staking and i % 10 == 0:
				betsize = max(bankroll * proportional_staking / 100, 5)
			x.append(i)
			result = slip_result[i]
			profit = (
				pay_general(slip_size, results=[result], payout=payout, normalize=True)
				* betsize
			)
			bankroll = bankroll + profit
			y.append(bankroll)

	return x, y, num_wins
