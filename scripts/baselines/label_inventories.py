"""scripts/baselines/label_inventories.py -- corpus-true (full-pool) closed label lists.

Fixes the "sample-observed closed lists (K4/K6/K7)" open scoping item from FREEZE_SHEET.md sect.
2 item 2: ``run_baseline._observed_label_set`` built its closed-choice label list from the UNION
of gold labels present in the CURRENT frozen dev(n=40)/test(n=60) draw only -- a rare label never
sampled was silently absent from the model's closed-choice prompt (never offered as an option, so
never answerable), and speech-massive's ``speaker_sex``/``speaker_age`` templates.py constants
were UNVERIFIED PLACEHOLDER GUESSES (not even sample-observed) before this fix.

Method: a metadata-only, no-audio-decode, full-pool scan (2026-07-09, WSL venv,
SPEECHRL_DATA_DIR=/mnt/e/chao_workspace/exploring-l4-intelligence/speechrl-data) --

  - slurp:                    repos/slurp/dataset/slurp/{train,devel,test}.jsonl, ALL THREE splits
                               combined (16,521 utterances) -- ``d["intent"]`` (already
                               "<scenario>_<action>") + ``d["entities"][*]["type"]``.
  - speech-massive fr-FR/de-DE: datasets/speech-massive/<locale>/validation-*.parquet, the FULL
                               validation split (2,033 rows/locale, every parquet row group read
                               via ``pyarrow.parquet.ParquetFile.read_row_group(columns=[...])`` --
                               intent_str/labels/speaker_sex/speaker_age columns only, audio NEVER
                               decoded). validation is the only split this grid's K5/K6/K7 cells
                               ever draw from (see run_baseline._load_rows), so a full scan of it
                               IS the corpus-true pool for this grid (no train-pool over-reach).
  - uro-bench UnderEmotion-en/-zh: datasets/uro-bench/UnderEmotion-{en,zh}/test-*.parquet, the
                               WHOLE (only) parquet split, ``emotion`` column only.

MAJOR FINDING (speaker_age): the templates.py placeholder ``SPEECH_MASSIVE_AGE_LABELS =
["Young Adult", "Adult", "Senior"]`` was not just incomplete but WRONG IN KIND -- the corpus
``speaker_age`` field holds individual integer ages as strings (e.g. "24", "35", ..., spanning
~20-74), not three coarse buckets. No bucketed value the old template offered ever matches the
real gold value, so a K5 speaker_age closed-choice cell run against the OLD template could not
score >0 by construction. Fixed here with the true ~29-value per-locale age list (and see
templates.py's ``_closed_options_block`` for the letters-vs-numbers rendering fix this forced --
these label sets exceed ``LETTERS``'s 8-slot lettered scheme, so >8-option lists now render as a
NUMBERED list, mirroring the K6 intent template's existing convention for large closed lists; see
``metrics._parse_choice``'s matching numbered-index parse branch).

Every constant below is SORTED (matches the ``_observed_label_set`` sort-then-list convention it
replaces) and is a plain list[str] -- drop-in for templates.py's K4/K5/K6/K7 builders. Per-dataset
provenance (source file(s)/query, row/utterance count scanned, per-label occurrence counts, and
the silent-miss delta against the CURRENT frozen dev+test sample) is documented in the comment
block directly above each constant.

Datasets NOT covered here (task scope: slurp, speech-massive fr-FR/de-DE, uro-bench
UnderEmotion-en/-zh only) keep the EXISTING sample-observed (``meta["_label_set"]``) or hardcoded
fallback in templates.py/metrics.py unchanged. See templates.py's K4/K5/K6/K7 branches in
``build_instruction`` (and the mirrored branches in ``metrics.score``) for exactly how this
module's constants take priority over that fallback, dataset-key by dataset-key.

2026-07-10 freeze-repair (wave-2 audit, ``vocalbench-emotion``): this dataset was NOT covered
above, and unlike slurp/speech-massive/uro-bench-UnderEmotion (all of which at least had a
sample-observed ``meta["_label_set"]`` populated by ``run_baseline._load_rows``), NO branch in
``_load_rows`` ever populated ``meta["_label_set"]`` for ``vocalbench-emotion`` at all -- it fell
straight through to the generic ``registry.LOADERS[dataset_key](split="test", ...)`` tail of
``_load_rows`` with no per-dataset post-processing. Effect: ``templates.build_instruction``'s K4
fallback (``meta.get("_label_set") or ["<observed set unavailable>"]``) rendered a ONE-OPTION
closed-choice prompt ("A. <observed set unavailable>") for every item, and ``metrics.score``'s
matching fallback (``meta.get("_label_set") or []``) scored against an EMPTY label set -- so
``score_k4_ser`` -> ``_closed_choice_score`` could never resolve ``match_idx`` (gold label is never
IN an empty list), making BOTH wave-2 ``vocalbench-emotion`` cells (dev n=40, test n=60) score
MECHANICALLY 0.0 regardless of the model's actual reply. See ``run_baseline._load_rows``'s new
``vocalbench-emotion`` branch (2026-07-10) for the fix -- it now populates
``meta["_label_set"] = VOCALBENCH_EMOTION_EMOTIONS`` (below) on every row, exactly the pattern the
uro-bench-UnderEmotion branch already used before ITS labels were promoted to a corpus-true
``K4_LABEL_SETS`` entry.
"""
from __future__ import annotations



# -----------------------------------------------------------------------------------------------
# slurp -- intent (scenario_action) set + slot-type set
# -----------------------------------------------------------------------------------------------


# SLURP_INTENTS -- 93 labels, corpus-true full-pool scan.
# source: repos/slurp/dataset/slurp/{train,devel,test}.jsonl (d['intent'])
# rows scanned: {'train': 11514, 'devel': 2033, 'test': 2974}
# (repos/slurp/dataset/slurp/{train,devel,test}.jsonl)
# silent-miss evidence (current dev n=40 seed=20260705 / test n=60 seed=20261705 draw, per
# run_baseline.py DEV_SEED/TEST_SEED): sample-observed union = 36/93 labels -- 57 labels the
# model would NEVER have seen in its closed-choice prompt were silently missing before this fix:
#   missing: addcontact, alarm_remove, audio_volume_mute, audio_volume_other, audio_volume_up,
#   cleaning, coffee, convert, cooking_query, createoradd, currency, definition, email_addcontact,
#   events, factoid, game, general_greet, general_joke, greet, hue_lightdim, hue_lightoff,
#   hue_lightup, iot_hue_lightdim, iot_hue_lighton, iot_wemo_off, iot_wemo_on, joke, likeness,
#   locations, music, play_game, podcasts, post, qa_definition, qa_stock, query, querycontact,
#   quirky, radio, recommendation_events, recommendation_locations, recommendation_movies, remove,
#   sendemail, set, settings, social_query, takeaway_order, takeaway_query, ticket, traffic,
#   transport_taxi, transport_ticket, transport_traffic, volume_other, wemo_off, wemo_on
# per-label counts (desc):
# calendar_set=1142 play_music=911 weather_query=834 general_quirky=817 calendar_query=783
# qa_factoid=765 news_query=704 email_query=604 email_sendemail=523 datetime_query=490
# calendar_remove=419 social_post=408 play_radio=394 qa_definition=378 cooking_recipe=320
# transport_query=314 lists_query=296 play_podcasts=282 recommendation_events=258 alarm_set=253
# lists_remove=251 lists_createoradd=234 recommendation_locations=234 play_audiobook=226
# music_query=215 qa_currency=208 iot_hue_lightoff=205 qa_stock=202 transport_ticket=186
# alarm_query=183 iot_hue_lightchange=183 takeaway_order=177 takeaway_query=176 iot_coffee=170
# email_querycontact=168 music_likeness=164 play_game=164 audio_volume_mute=157
# transport_traffic=152 transport_taxi=150 social_query=149 audio_volume_up=135 iot_cleaning=135
# qa_maths=116 alarm_remove=113 iot_hue_lightdim=111 iot_hue_lightup=111
# recommendation_movies=102 general_joke=101 datetime_convert=75 iot_wemo_off=72
# audio_volume_down=71 email_addcontact=70 query=68 iot_wemo_on=64 music_settings=64
# iot_hue_lighton=30 music=27 general_greet=25 audio_volume_other=23 music_dislikeness=20
# quirky=12 factoid=10 remove=9 sendemail=9 set=9 hue_lightoff=8 podcasts=8 createoradd=7
# radio=7 cooking_query=6 post=6 currency=5 game=5 joke=5 coffee=4 hue_lightup=4 cleaning=3
# greet=3 hue_lightdim=3 wemo_off=3 traffic=2 addcontact=1 convert=1 definition=1 events=1
# likeness=1 locations=1 querycontact=1 settings=1 ticket=1 volume_other=1 wemo_on=1
SLURP_INTENTS = [
    'addcontact', 'alarm_query', 'alarm_remove', 'alarm_set', 'audio_volume_down',
    'audio_volume_mute', 'audio_volume_other', 'audio_volume_up', 'calendar_query',
    'calendar_remove', 'calendar_set', 'cleaning', 'coffee', 'convert', 'cooking_query',
    'cooking_recipe', 'createoradd', 'currency', 'datetime_convert', 'datetime_query',
    'definition', 'email_addcontact', 'email_query', 'email_querycontact', 'email_sendemail',
    'events', 'factoid', 'game', 'general_greet', 'general_joke', 'general_quirky', 'greet',
    'hue_lightdim', 'hue_lightoff', 'hue_lightup', 'iot_cleaning', 'iot_coffee',
    'iot_hue_lightchange', 'iot_hue_lightdim', 'iot_hue_lightoff', 'iot_hue_lighton',
    'iot_hue_lightup', 'iot_wemo_off', 'iot_wemo_on', 'joke', 'likeness', 'lists_createoradd',
    'lists_query', 'lists_remove', 'locations', 'music', 'music_dislikeness', 'music_likeness',
    'music_query', 'music_settings', 'news_query', 'play_audiobook', 'play_game', 'play_music',
    'play_podcasts', 'play_radio', 'podcasts', 'post', 'qa_currency', 'qa_definition',
    'qa_factoid', 'qa_maths', 'qa_stock', 'query', 'querycontact', 'quirky', 'radio',
    'recommendation_events', 'recommendation_locations', 'recommendation_movies', 'remove',
    'sendemail', 'set', 'settings', 'social_post', 'social_query', 'takeaway_order',
    'takeaway_query', 'ticket', 'traffic', 'transport_query', 'transport_taxi',
    'transport_ticket', 'transport_traffic', 'volume_other', 'weather_query', 'wemo_off',
    'wemo_on',
]


# SLURP_SLOT_TYPES -- 55 labels, corpus-true full-pool scan.
# source: repos/slurp/dataset/slurp/{train,devel,test}.jsonl (d['entities'][*]['type'])
# rows scanned: {'train': 11514, 'devel': 2033, 'test': 2974}
# (repos/slurp/dataset/slurp/{train,devel,test}.jsonl)
# silent-miss evidence (current dev n=40 seed=20260705 / test n=60 seed=20261705 draw, per
# run_baseline.py DEV_SEED/TEST_SEED): sample-observed union = 26/55 labels -- 29 labels the
# model would NEVER have seen in its closed-choice prompt were silently missing before this fix:
#   missing: alarm_type, app_name, audiobook_author, business_type, change_amount, coffee_type,
#   currency_name, definition_word, drink_type, email_address, email_folder, game_name, game_type,
#   general_frequency, ingredient, joke_type, meal_type, movie_name, movie_type, music_album,
#   music_descriptor, music_genre, order_type, personal_info, podcast_name, song_name, sport_type,
#   transport_agency, transport_descriptor
# per-label counts (desc):
# date=2585 place_name=1579 event_name=1421 person=1219 time=1130 media_type=705
# business_name=534 weather_descriptor=461 transport_type=437 food_type=418 relation=352
# timeofday=348 artist_name=339 definition_word=319 device_type=318 currency_name=316
# list_name=288 house_place=281 news_topic=272 music_genre=269 business_type=254
# player_setting=237 song_name=190 radio_name=186 order_type=145 color_type=142 game_name=131
# general_frequency=119 audiobook_name=112 podcast_descriptor=102 personal_info=98 meal_type=95
# playlist_name=85 podcast_name=81 time_zone=74 app_name=64 change_amount=63 music_descriptor=60
# joke_type=54 transport_agency=49 email_address=44 email_folder=43 ingredient=30 coffee_type=28
# cooking_type=27 movie_name=20 transport_name=15 alarm_type=14 movie_type=14 drink_type=13
# audiobook_author=12 transport_descriptor=10 sport_type=6 game_type=2 music_album=2
SLURP_SLOT_TYPES = [
    'alarm_type', 'app_name', 'artist_name', 'audiobook_author', 'audiobook_name',
    'business_name', 'business_type', 'change_amount', 'coffee_type', 'color_type',
    'cooking_type', 'currency_name', 'date', 'definition_word', 'device_type', 'drink_type',
    'email_address', 'email_folder', 'event_name', 'food_type', 'game_name', 'game_type',
    'general_frequency', 'house_place', 'ingredient', 'joke_type', 'list_name', 'meal_type',
    'media_type', 'movie_name', 'movie_type', 'music_album', 'music_descriptor', 'music_genre',
    'news_topic', 'order_type', 'person', 'personal_info', 'place_name', 'player_setting',
    'playlist_name', 'podcast_descriptor', 'podcast_name', 'radio_name', 'relation', 'song_name',
    'sport_type', 'time', 'time_zone', 'timeofday', 'transport_agency', 'transport_descriptor',
    'transport_name', 'transport_type', 'weather_descriptor',
]


# -----------------------------------------------------------------------------------------------
# speech-massive fr-FR -- intent_str set + slot label set + speaker_sex/speaker_age
# -----------------------------------------------------------------------------------------------


# SPEECH_MASSIVE_FR_FR_INTENTS -- 59 labels, corpus-true full-pool scan.
# source: datasets/speech-massive/fr-FR/validation-*.parquet (rec['intent_str'])
# rows scanned: 2033 (datasets/speech-massive/fr-FR/validation-*.parquet (full split, 2
# shard(s), 22 row groups))
# silent-miss evidence (current dev n=40 seed=20260705 / test n=60 seed=20261705 draw, per
# run_baseline.py DEV_SEED/TEST_SEED): sample-observed union = 37/59 labels -- 22 labels the
# model would NEVER have seen in its closed-choice prompt were silently missing before this fix:
#   missing: alarm_remove, alarm_set, audio_volume_down, calendar_remove, cooking_query,
#   email_addcontact, general_greet, iot_hue_lightchange, iot_hue_lightup, iot_wemo_off,
#   iot_wemo_on, lists_createoradd, lists_remove, music_dislikeness, music_likeness,
#   music_settings, play_game, qa_maths, recommendation_locations, social_query, takeaway_order,
#   transport_ticket
# per-label counts (desc):
# calendar_set=131 weather_query=126 play_music=123 general_quirky=105 calendar_query=102
# qa_factoid=90 news_query=82 email_query=73 datetime_query=64 email_sendemail=63
# qa_definition=55 lists_query=50 social_post=50 calendar_remove=47 play_radio=46
# cooking_recipe=41 lists_remove=37 transport_query=36 play_audiobook=35 play_podcasts=34
# qa_currency=32 alarm_set=31 recommendation_locations=31 music_query=30 transport_taxi=27
# recommendation_events=26 lists_createoradd=25 transport_ticket=25 qa_stock=24
# takeaway_query=24 iot_hue_lightchange=22 play_game=22 transport_traffic=22 takeaway_order=20
# alarm_query=19 iot_cleaning=19 social_query=18 iot_hue_lightdim=17 iot_hue_lightoff=17
# email_querycontact=16 music_likeness=16 audio_volume_mute=15 general_joke=15 alarm_remove=14
# iot_coffee=14 qa_maths=13 audio_volume_up=12 iot_hue_lightup=12 recommendation_movies=12
# datetime_convert=9 audio_volume_down=8 music_settings=8 iot_wemo_on=7 email_addcontact=5
# iot_hue_lighton=5 iot_wemo_off=5 cooking_query=2 general_greet=2 music_dislikeness=2
SPEECH_MASSIVE_FR_FR_INTENTS = [
    'alarm_query', 'alarm_remove', 'alarm_set', 'audio_volume_down', 'audio_volume_mute',
    'audio_volume_up', 'calendar_query', 'calendar_remove', 'calendar_set', 'cooking_query',
    'cooking_recipe', 'datetime_convert', 'datetime_query', 'email_addcontact', 'email_query',
    'email_querycontact', 'email_sendemail', 'general_greet', 'general_joke', 'general_quirky',
    'iot_cleaning', 'iot_coffee', 'iot_hue_lightchange', 'iot_hue_lightdim', 'iot_hue_lightoff',
    'iot_hue_lighton', 'iot_hue_lightup', 'iot_wemo_off', 'iot_wemo_on', 'lists_createoradd',
    'lists_query', 'lists_remove', 'music_dislikeness', 'music_likeness', 'music_query',
    'music_settings', 'news_query', 'play_audiobook', 'play_game', 'play_music', 'play_podcasts',
    'play_radio', 'qa_currency', 'qa_definition', 'qa_factoid', 'qa_maths', 'qa_stock',
    'recommendation_events', 'recommendation_locations', 'recommendation_movies', 'social_post',
    'social_query', 'takeaway_order', 'takeaway_query', 'transport_query', 'transport_taxi',
    'transport_ticket', 'transport_traffic', 'weather_query',
]


# SPEECH_MASSIVE_FR_FR_SLOT_LABELS -- 52 labels, corpus-true full-pool scan.
# source: datasets/speech-massive/fr-FR/validation-*.parquet (rec['labels'], 'Other' excluded --
# matches run_baseline._load_rows's existing filter)
# rows scanned: 2033 (datasets/speech-massive/fr-FR/validation-*.parquet (full split, 2
# shard(s), 22 row groups))
# silent-miss evidence (current dev n=40 seed=20260705 / test n=60 seed=20261705 draw, per
# run_baseline.py DEV_SEED/TEST_SEED): sample-observed union = 30/52 labels -- 22 labels the
# model would NEVER have seen in its closed-choice prompt were silently missing before this fix:
#   missing: audiobook_author, audiobook_name, change_amount, coffee_type, color_type,
#   cooking_type, drink_type, email_address, email_folder, game_name, game_type, meal_type,
#   movie_name, movie_type, order_type, personal_info, podcast_descriptor, podcast_name,
#   radio_name, sport_type, time_zone, transport_name
# per-label counts (desc):
# date=531 time=360 place_name=294 event_name=252 person=177 media_type=143 food_type=105
# business_name=92 news_topic=90 definition_word=72 device_type=69 weather_descriptor=69
# timeofday=67 currency_name=66 artist_name=65 list_name=64 relation=54 transport_type=54
# player_setting=50 radio_name=49 business_type=46 music_genre=46 audiobook_name=44 song_name=40
# general_frequency=33 house_place=33 game_name=30 order_type=30 podcast_descriptor=23
# playlist_name=20 podcast_name=20 music_descriptor=19 email_folder=18 personal_info=18
# color_type=17 email_address=15 meal_type=12 change_amount=11 joke_type=9 transport_agency=9
# app_name=8 ingredient=8 drink_type=5 time_zone=5 movie_name=3 sport_type=3 transport_name=3
# audiobook_author=2 coffee_type=2 cooking_type=2 movie_type=2 game_type=1
SPEECH_MASSIVE_FR_FR_SLOT_LABELS = [
    'app_name', 'artist_name', 'audiobook_author', 'audiobook_name', 'business_name',
    'business_type', 'change_amount', 'coffee_type', 'color_type', 'cooking_type',
    'currency_name', 'date', 'definition_word', 'device_type', 'drink_type', 'email_address',
    'email_folder', 'event_name', 'food_type', 'game_name', 'game_type', 'general_frequency',
    'house_place', 'ingredient', 'joke_type', 'list_name', 'meal_type', 'media_type',
    'movie_name', 'movie_type', 'music_descriptor', 'music_genre', 'news_topic', 'order_type',
    'person', 'personal_info', 'place_name', 'player_setting', 'playlist_name',
    'podcast_descriptor', 'podcast_name', 'radio_name', 'relation', 'song_name', 'sport_type',
    'time', 'time_zone', 'timeofday', 'transport_agency', 'transport_name', 'transport_type',
    'weather_descriptor',
]


# SPEECH_MASSIVE_FR_FR_SPEAKER_SEX -- 3 labels, corpus-true full-pool scan.
# source: datasets/speech-massive/fr-FR/validation-*.parquet (rec['speaker_sex'])
# rows scanned: 2033 (datasets/speech-massive/fr-FR/validation-*.parquet (full split, 2
# shard(s), 22 row groups))
# silent-miss evidence (current dev n=40 seed=20260705 / test n=60 seed=20261705 draw, per
# run_baseline.py DEV_SEED/TEST_SEED): sample-observed union = 3/3 labels -- 0 labels the model
# would NEVER have seen in its closed-choice prompt were silently missing before this fix:
#   missing: (none -- full pool already covered by the dev+test draw)
# per-label counts (desc):
# Male=998 Female=994 Unidentified=41
SPEECH_MASSIVE_FR_FR_SPEAKER_SEX = [
    'Female', 'Male', 'Unidentified',
]


# SPEECH_MASSIVE_FR_FR_SPEAKER_AGE -- 29 labels, corpus-true full-pool scan.
# source: datasets/speech-massive/fr-FR/validation-*.parquet (rec['speaker_age'])
# rows scanned: 2033 (datasets/speech-massive/fr-FR/validation-*.parquet (full split, 2
# shard(s), 22 row groups))
# silent-miss evidence (current dev n=40 seed=20260705 / test n=60 seed=20261705 draw, per
# run_baseline.py DEV_SEED/TEST_SEED): sample-observed union = 28/29 labels -- 1 labels the
# model would NEVER have seen in its closed-choice prompt were silently missing before this fix:
#   missing: 42
# NOTE: individual integer ages as strings, NOT buckets -- see module docstring's MAJOR FINDING.
# per-label counts (desc):
# 24=234 25=117 22=116 32=116 47=110 27=108 40=103 37=102 33=80 30=79 Unidentified=78 31=74
# 21=69 39=67 43=50 67=46 60=44 26=40 34=40 42=40 46=40 28=39 35=39 48=39 55=38 38=35 68=35
# 74=29 50=26
SPEECH_MASSIVE_FR_FR_SPEAKER_AGE = [
    '21', '22', '24', '25', '26', '27', '28', '30', '31', '32', '33', '34', '35', '37', '38',
    '39', '40', '42', '43', '46', '47', '48', '50', '55', '60', '67', '68', '74', 'Unidentified',
]


# -----------------------------------------------------------------------------------------------
# speech-massive de-DE -- intent_str set + slot label set + speaker_sex/speaker_age
# -----------------------------------------------------------------------------------------------


# SPEECH_MASSIVE_DE_DE_INTENTS -- 59 labels, corpus-true full-pool scan.
# source: datasets/speech-massive/de-DE/validation-*.parquet (rec['intent_str'])
# rows scanned: 2033 (datasets/speech-massive/de-DE/validation-*.parquet (full split, 2
# shard(s), 22 row groups))
# silent-miss evidence (current dev n=40 seed=20260705 / test n=60 seed=20261705 draw, per
# run_baseline.py DEV_SEED/TEST_SEED): sample-observed union = 42/59 labels -- 17 labels the
# model would NEVER have seen in its closed-choice prompt were silently missing before this fix:
#   missing: alarm_remove, audio_volume_down, audio_volume_up, cooking_query, datetime_convert,
#   general_greet, iot_coffee, iot_hue_lightdim, iot_hue_lighton, iot_wemo_off, lists_query,
#   music_dislikeness, music_query, music_settings, play_game, transport_query, transport_traffic
# per-label counts (desc):
# calendar_set=131 weather_query=126 play_music=123 general_quirky=105 calendar_query=102
# qa_factoid=90 news_query=82 email_query=73 datetime_query=64 email_sendemail=63
# qa_definition=55 lists_query=50 social_post=50 calendar_remove=47 play_radio=46
# cooking_recipe=41 lists_remove=37 transport_query=36 play_audiobook=35 play_podcasts=34
# qa_currency=32 alarm_set=31 recommendation_locations=31 music_query=30 transport_taxi=27
# recommendation_events=26 lists_createoradd=25 transport_ticket=25 qa_stock=24
# takeaway_query=24 iot_hue_lightchange=22 play_game=22 transport_traffic=22 takeaway_order=20
# alarm_query=19 iot_cleaning=19 social_query=18 iot_hue_lightdim=17 iot_hue_lightoff=17
# email_querycontact=16 music_likeness=16 audio_volume_mute=15 general_joke=15 alarm_remove=14
# iot_coffee=14 qa_maths=13 audio_volume_up=12 iot_hue_lightup=12 recommendation_movies=12
# datetime_convert=9 audio_volume_down=8 music_settings=8 iot_wemo_on=7 email_addcontact=5
# iot_hue_lighton=5 iot_wemo_off=5 cooking_query=2 general_greet=2 music_dislikeness=2
SPEECH_MASSIVE_DE_DE_INTENTS = [
    'alarm_query', 'alarm_remove', 'alarm_set', 'audio_volume_down', 'audio_volume_mute',
    'audio_volume_up', 'calendar_query', 'calendar_remove', 'calendar_set', 'cooking_query',
    'cooking_recipe', 'datetime_convert', 'datetime_query', 'email_addcontact', 'email_query',
    'email_querycontact', 'email_sendemail', 'general_greet', 'general_joke', 'general_quirky',
    'iot_cleaning', 'iot_coffee', 'iot_hue_lightchange', 'iot_hue_lightdim', 'iot_hue_lightoff',
    'iot_hue_lighton', 'iot_hue_lightup', 'iot_wemo_off', 'iot_wemo_on', 'lists_createoradd',
    'lists_query', 'lists_remove', 'music_dislikeness', 'music_likeness', 'music_query',
    'music_settings', 'news_query', 'play_audiobook', 'play_game', 'play_music', 'play_podcasts',
    'play_radio', 'qa_currency', 'qa_definition', 'qa_factoid', 'qa_maths', 'qa_stock',
    'recommendation_events', 'recommendation_locations', 'recommendation_movies', 'social_post',
    'social_query', 'takeaway_order', 'takeaway_query', 'transport_query', 'transport_taxi',
    'transport_ticket', 'transport_traffic', 'weather_query',
]


# SPEECH_MASSIVE_DE_DE_SLOT_LABELS -- 53 labels, corpus-true full-pool scan.
# source: datasets/speech-massive/de-DE/validation-*.parquet (rec['labels'], 'Other' excluded --
# matches run_baseline._load_rows's existing filter)
# rows scanned: 2033 (datasets/speech-massive/de-DE/validation-*.parquet (full split, 2
# shard(s), 22 row groups))
# silent-miss evidence (current dev n=40 seed=20260705 / test n=60 seed=20261705 draw, per
# run_baseline.py DEV_SEED/TEST_SEED): sample-observed union = 34/53 labels -- 19 labels the
# model would NEVER have seen in its closed-choice prompt were silently missing before this fix:
#   missing: alarm_type, change_amount, coffee_type, cooking_type, definition_word, drink_type,
#   email_folder, game_name, game_type, general_frequency, ingredient, movie_name, movie_type,
#   music_descriptor, playlist_name, radio_name, sport_type, time_zone, transport_name
# per-label counts (desc):
# date=475 time=281 place_name=272 event_name=194 person=180 media_type=125 business_name=81
# currency_name=78 artist_name=72 news_topic=72 food_type=69 weather_descriptor=66 timeofday=63
# radio_name=55 definition_word=54 transport_type=54 player_setting=53 song_name=53 list_name=52
# device_type=50 music_genre=50 relation=49 business_type=37 audiobook_name=36 order_type=31
# general_frequency=28 game_name=27 house_place=25 podcast_descriptor=24 email_address=18
# color_type=17 playlist_name=13 podcast_name=13 meal_type=12 music_descriptor=12
# change_amount=11 personal_info=11 app_name=9 joke_type=9 transport_agency=9 email_folder=6
# ingredient=5 time_zone=5 transport_name=5 drink_type=4 alarm_type=3 movie_name=3
# audiobook_author=2 coffee_type=2 cooking_type=2 movie_type=2 game_type=1 sport_type=1
SPEECH_MASSIVE_DE_DE_SLOT_LABELS = [
    'alarm_type', 'app_name', 'artist_name', 'audiobook_author', 'audiobook_name',
    'business_name', 'business_type', 'change_amount', 'coffee_type', 'color_type',
    'cooking_type', 'currency_name', 'date', 'definition_word', 'device_type', 'drink_type',
    'email_address', 'email_folder', 'event_name', 'food_type', 'game_name', 'game_type',
    'general_frequency', 'house_place', 'ingredient', 'joke_type', 'list_name', 'meal_type',
    'media_type', 'movie_name', 'movie_type', 'music_descriptor', 'music_genre', 'news_topic',
    'order_type', 'person', 'personal_info', 'place_name', 'player_setting', 'playlist_name',
    'podcast_descriptor', 'podcast_name', 'radio_name', 'relation', 'song_name', 'sport_type',
    'time', 'time_zone', 'timeofday', 'transport_agency', 'transport_name', 'transport_type',
    'weather_descriptor',
]


# SPEECH_MASSIVE_DE_DE_SPEAKER_SEX -- 2 labels, corpus-true full-pool scan.
# source: datasets/speech-massive/de-DE/validation-*.parquet (rec['speaker_sex'])
# rows scanned: 2033 (datasets/speech-massive/de-DE/validation-*.parquet (full split, 2
# shard(s), 22 row groups))
# silent-miss evidence (current dev n=40 seed=20260705 / test n=60 seed=20261705 draw, per
# run_baseline.py DEV_SEED/TEST_SEED): sample-observed union = 2/2 labels -- 0 labels the model
# would NEVER have seen in its closed-choice prompt were silently missing before this fix:
#   missing: (none -- full pool already covered by the dev+test draw)
# per-label counts (desc):
# Male=1082 Female=951
SPEECH_MASSIVE_DE_DE_SPEAKER_SEX = [
    'Female', 'Male',
]


# SPEECH_MASSIVE_DE_DE_SPEAKER_AGE -- 29 labels, corpus-true full-pool scan.
# source: datasets/speech-massive/de-DE/validation-*.parquet (rec['speaker_age'])
# rows scanned: 2033 (datasets/speech-massive/de-DE/validation-*.parquet (full split, 2
# shard(s), 22 row groups))
# silent-miss evidence (current dev n=40 seed=20260705 / test n=60 seed=20261705 draw, per
# run_baseline.py DEV_SEED/TEST_SEED): sample-observed union = 28/29 labels -- 1 labels the
# model would NEVER have seen in its closed-choice prompt were silently missing before this fix:
#   missing: 47
# NOTE: individual integer ages as strings, NOT buckets -- see module docstring's MAJOR FINDING.
# per-label counts (desc):
# 35=253 29=150 32=135 43=120 22=100 23=100 34=99 31=97 40=97 27=80 39=78 24=60 25=60 26=60
# 64=60 56=49 30=46 28=40 33=40 37=40 46=39 21=38 58=37 53=35 48=34 20=32 49=19 72=19 47=16
SPEECH_MASSIVE_DE_DE_SPEAKER_AGE = [
    '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
    '35', '37', '39', '40', '43', '46', '47', '48', '49', '53', '56', '58', '64', '72',
]


# -----------------------------------------------------------------------------------------------
# uro-bench UnderEmotion-en -- emotion set (whole parquet, no sampling)
# -----------------------------------------------------------------------------------------------


# URO_BENCH_UNDEREMOTION_EN_EMOTIONS -- 41 labels, corpus-true full-pool scan.
# source: datasets/uro-bench/UnderEmotion-en/test-*.parquet (rec['emotion'])
# rows scanned: 137 (datasets/uro-bench/UnderEmotion-en/test-*.parquet (full split, 1 shard(s)))
# silent-miss evidence (current dev n=40 seed=20260705 / test n=60 seed=20261705 draw, per
# run_baseline.py DEV_SEED/TEST_SEED): sample-observed union = 35/41 labels -- 6 labels the
# model would NEVER have seen in its closed-choice prompt were silently missing before this fix:
#   missing: Annoyed, Happy, Irritated, Joyful, Optimistic, Unmotivated
# per-label counts (desc):
# happy=15 angry=14 fearful=13 surprised=13 sad=11 Frustrated=8 Confused=5 Surprised=4
# Disappointed=3 Exhausted=3 Optimistic=3 Proud=3 Relieved=3 Uncertain=3 Annoyed=2 Excited=2
# Grateful=2 Impatient=2 Irritated=2 Overwhelmed=2 Regretful=2 Shocked=2 Stressed=2 Amazed=1
# Anxious=1 Apprehensive=1 Content=1 Distrustful=1 Eager=1 Happy=1 Helpless=1 Hopeful=1
# Hopeless=1 Inadequate=1 Indifferent=1 Joyful=1 Nostalgic=1 Resigned=1 Resilient=1 Tired=1
# Unmotivated=1
URO_BENCH_UNDEREMOTION_EN_EMOTIONS = [
    'Amazed', 'Annoyed', 'Anxious', 'Apprehensive', 'Confused', 'Content', 'Disappointed',
    'Distrustful', 'Eager', 'Excited', 'Exhausted', 'Frustrated', 'Grateful', 'Happy',
    'Helpless', 'Hopeful', 'Hopeless', 'Impatient', 'Inadequate', 'Indifferent', 'Irritated',
    'Joyful', 'Nostalgic', 'Optimistic', 'Overwhelmed', 'Proud', 'Regretful', 'Relieved',
    'Resigned', 'Resilient', 'Shocked', 'Stressed', 'Surprised', 'Tired', 'Uncertain',
    'Unmotivated', 'angry', 'fearful', 'happy', 'sad', 'surprised',
]


# -----------------------------------------------------------------------------------------------
# uro-bench UnderEmotion-zh -- emotion set (whole parquet, no sampling)
# -----------------------------------------------------------------------------------------------


# URO_BENCH_UNDEREMOTION_ZH_EMOTIONS -- 49 labels, corpus-true full-pool scan.
# source: datasets/uro-bench/UnderEmotion-zh/test-*.parquet (rec['emotion'])
# rows scanned: 79 (datasets/uro-bench/UnderEmotion-zh/test-*.parquet (full split, 1 shard(s)))
# silent-miss evidence (current dev n=40 seed=20260705 / test n=60 seed=20261705 draw, per
# run_baseline.py DEV_SEED/TEST_SEED): sample-observed union = 45/49 labels -- 4 labels the
# model would NEVER have seen in its closed-choice prompt were silently missing before this fix:
#   missing: 后悔, 惊讶的, 渴望, 绝望的
# per-label counts (desc):
# 沮丧=6 惊讶=5 困惑=4 不确定=3 乐观的=3 压力大=3 不知所措的=2 兴奋=2 失望=2 宽慰的=2 恼火=2 感激的=2 有压力=2 沮丧的=2 烦恼=2 疲惫=2
# 自豪的=2 震惊=2 不信任的=1 不充分的=1 不耐烦=1 不耐烦的=1 乐观=1 充满希望的=1 兴奋的=1 冷漠=1 后悔=1 困惑的=1 坚韧的=1 失望的=1 快乐=1
# 怀念的=1 惊讶的=1 愉快=1 感激=1 担忧的=1 无助的=1 松了口气=1 渴望=1 满足=1 满足的=1 焦虑的=1 疲倦=1 精疲力尽的=1 绝望的=1 缺乏动力的=1 自豪=1
# 遗憾=1 顺从=1
URO_BENCH_UNDEREMOTION_ZH_EMOTIONS = [
    '不信任的', '不充分的', '不知所措的', '不确定', '不耐烦', '不耐烦的', '乐观', '乐观的', '充满希望的', '兴奋', '兴奋的', '冷漠',
    '压力大', '后悔', '困惑', '困惑的', '坚韧的', '失望', '失望的', '宽慰的', '快乐', '怀念的', '恼火', '惊讶', '惊讶的', '愉快',
    '感激', '感激的', '担忧的', '无助的', '有压力', '松了口气', '沮丧', '沮丧的', '渴望', '满足', '满足的', '烦恼', '焦虑的', '疲倦',
    '疲惫', '精疲力尽的', '绝望的', '缺乏动力的', '自豪', '自豪的', '遗憾', '震惊', '顺从',
]


# -----------------------------------------------------------------------------------------------
# vocalbench-emotion -- Question_emo set (whole parquet, no sampling)
# -----------------------------------------------------------------------------------------------


# VOCALBENCH_EMOTION_EMOTIONS -- 5 labels, corpus-true full-pool scan (2026-07-10 freeze-repair,
# wave-2 audit -- see module docstring's dedicated note above for how this dataset was completely
# unwired before this fix, not merely sample-observed-vs-full-pool undercovered like the other
# entries in this module).
# source: datasets/vocalbench/parquet/emotion.parquet (rec['Question_emo'])
# rows scanned: 500 (datasets/vocalbench/parquet/emotion.parquet, the WHOLE (only) 'test' split --
# see scripts/loaders/vocalbench.py's load_vocalbench_emotion / _require_test_split)
# silent-miss evidence: N/A in the usual sense -- this is a small, fully-populated 5-way vocabulary
# (100 rows/label out of 500, verified below), not a long-tail distribution a 40/60-row sample
# could plausibly under-cover; the CURRENT frozen dev(n=40)/test(n=60) draws' gold-label union is
# in fact already the full 5/5 (see the per-item "gold_label" values recorded in the pre-fix
# _repro/baselines/vocalbench-emotion__*__{dev,test}.json -- angry/happy/neutral/sad/surprised all
# present in both splits). The actual defect this fixes is categorically different: NO branch in
# run_baseline._load_rows ever populated meta["_label_set"] for this dataset key AT ALL (see module
# docstring) -- the model was never shown any real options, not merely a truncated set.
# per-label counts (exact, exhaustive):
# angry=100 happy=100 neutral=100 sad=100 surprised=100
VOCALBENCH_EMOTION_EMOTIONS = [
    'angry', 'happy', 'neutral', 'sad', 'surprised',
]
