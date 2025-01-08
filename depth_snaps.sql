with ddepth as (select d.*, s.pfr_player_id from depth d join full_ids s
on d.gsis_id = s.gsis_id 
)
select * from ddepth dd
join snaps s ON
dd.pfr_player_id = s.pfr_player_id
and dd.season = s.season
and dd.week = s.week
;