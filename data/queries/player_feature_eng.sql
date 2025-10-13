with base as (
select
depth.dense_depth,depth.position, depth.gsis_id,coalesce(weekly.player_id, depth.gsis_id) as player_id, depth.game_id,depth.season,
depth.week, depth.player_name,coalesce(weekly.player_display_name, depth.player_name) as player_display_name,
completions, attempts,
       passing_yards, passing_tds, passing_interceptions, sacks_suffered, sack_yards_lost,
       sack_fumbles, sack_fumbles_lost, passing_air_yards,
       passing_yards_after_catch, passing_first_downs, passing_epa,
       passing_2pt_conversions, pacr, carries, rushing_yards,
       rushing_tds, rushing_fumbles, rushing_fumbles_lost,
       rushing_first_downs, rushing_epa, rushing_2pt_conversions,
       receptions, targets, receiving_yards, receiving_tds,
       receiving_fumbles, receiving_fumbles_lost, receiving_air_yards,
       receiving_yards_after_catch, receiving_first_downs, receiving_epa,
       receiving_2pt_conversions, racr, target_share, air_yards_share,
       wopr, special_teams_tds, fantasy_points, fantasy_points_ppr
from weekly full join depth
on weekly.game_id = depth.game_id
and weekly.player_id = depth.gsis_id
where depth.formation = 'Offense'
),
snps as (
select ids.gsis_id, 
snaps.game_id, snaps.team, snaps.opponent, snaps.offense_snaps, snaps.offense_pct
  from ids
join snaps on ids.pfr_id = snaps.pfr_player_id
)
select * 
from base left join snps using(gsis_id, game_id)
--where game_id = '2024_04_TEN_MIA'
--and position = 'QB'
order by base.season,base.week
;