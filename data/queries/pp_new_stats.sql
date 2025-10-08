with pids as (select name,gsis_id from ids)
,
rushing as (
    select
        game_id,
        gsis_id,
        rusher_player_name as pbp_name,
        game_seconds_remaining,
        pids.name,
        rushing_yards,
        row_number()
            over (
                partition by game_id, gsis_id
                order by game_seconds_remaining desc
            )
            as row_num
    from pbp left join pids
    on pbp.rusher_player_id = pids.gsis_id
    where
        season > 2022
),

receiving as (
    select
        game_id,
        gsis_id,
        receiver_player_name as pbp_name,
        game_seconds_remaining,
        pids.name,
        receiving_yards,
        row_number()
            over (
                partition by game_id, gsis_id
                order by game_seconds_remaining desc
            )
            as row_num
    from pbp left join pids on pbp.receiver_player_id = pids.gsis_id
    where
        complete_pass = 1
        and season > 2022

),

rushing_agg as (
    select distinct
        game_id,
        gsis_id,
        pbp_name,
        name,
        first(rushing_yards ignore nulls)
            over (
                partition by gsis_id, game_id
                order by game_seconds_remaining desc
            )
            as first_rush,
        sum(case when row_num <= 5 then rushing_yards else 0 end)
            over (partition by gsis_id, game_id)
        *
        max(case when row_num >= 5 then 1 end)
            over (partition by gsis_id, game_id) 
        as yards_5_rushes,
        max(rushing_yards) over (partition by gsis_id, game_id) as longest_rush,
        first(rushing_yards) over (partition by gsis_id, game_id) as yards_first_rush
    from rushing
    where gsis_id is not null
),

receiving_agg as (
    select distinct
        game_id,
        gsis_id,
        pbp_name,
        name,
        first(receiving_yards ignore nulls)
            over (
                partition by gsis_id, game_id
                order by game_seconds_remaining desc
            )
            as yards_first_catch,
        sum(case when row_num <= 2 then receiving_yards else 0 end)
            over (partition by gsis_id, game_id)
        *
        max(case when row_num >= 2 then 1 end)
            over (partition by gsis_id, game_id) 
        as yards_2_catches,
        max(receiving_yards) over (partition by gsis_id, game_id) as longest_reception
    from receiving
    ),
    rr_agg as (
    select * from receiving_agg full join rushing_agg
    using (
            game_id,
        gsis_id,
        pbp_name,
        name
        )
        ),
quarters as (
    select
        game_id,
        gsis_id,
        qtr,
        pids.name,
        coalesce(sum(rushing_yards), 0) as rushing_yards,
        coalesce(sum(receiving_yards), 0) as receiving_yards,
        coalesce(sum(case when complete_pass then 1 else 0 end)) as receptions
    from pbp
    left join
        pids
        on coalesce(pbp.rusher_player_id, pbp.receiver_player_id) = pids.gsis_id
    where
        season > 2022
    group by all
),
quarters_agg as (
    select
        game_id,
        gsis_id,
        name,
        sum(case when rushing_yards > 5 then 1 else 0 end)
            as qtrs_over_5_rush_yards,
        sum(case when receiving_yards > 10 then 1 else 0 end)
            as qtrs_over_5_rec_yards,
        sum(case when receptions > 0 then 1 else 0 end) as qtrs_with_reception
    from quarters
    group by all
)

select * from rr_agg left join quarters_agg using (game_id, gsis_id, name)
right join weekly ON
weekly.game_id = rr_agg.game_id
and weekly.player_id = rr_agg.gsis_id
join schedule on weekly.game_id = schedule.game_id
where schedule.season > 2020
;