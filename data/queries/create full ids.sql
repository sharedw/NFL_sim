-- SQLite
--drop table if exists full_ids;
--create table full_ids as 
with roster_ids as (
select 
lower(replace(REPLACE(player_name, ' Jr.', ''), 'Mike Woods', 'Michael Woods II' )) as player
, player_id as gsis_id
, team
, week
, season
 from rosters
),
snaps_ids as (
select 
lower(snaps.player) as player,
snaps.team,
snaps.week,
snaps.season,
snaps.position,
snaps.pfr_player_id,
ids.gsis_id
from snaps  join ids on snaps.pfr_player_id = ids.pfr_id
),
snaps_plus_roster as (
select 
si.player,
si.team,
si.week,
si.season,
si.position,
si.pfr_player_id,
coalesce(si.gsis_id, r.gsis_id) as gsis_id
from snaps_ids si left join roster_ids r
on si.player = r.player
and si.team = r.team
and si.week = r.week
and si.season = r.season
)
select 
distinct player, pfr_player_id, gsis_id from snaps_plus_roster

;
-- was the problem missing pfr ids? check both ways