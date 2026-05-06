#!/usr/bin/env python3
"""
NBA Data Scraper v2 — 5 Seasons (2021-22 through 2025-26)
==========================================================
Collects player per-game stats, advanced stats, team standings,
and game-level box scores using BULK endpoints that return an
entire season per API call.

Key change from v1: box scores now use LeagueGameLog (player + team)
which returns all data in ~10 total API calls instead of ~12,000
individual BoxScore calls. This eliminates the rate-limit death
spiral that made v1 unusable for overnight runs.

Outputs:
    player_pergame.csv   - per-game averages for every player
    player_advanced.csv  - advanced metrics (TS%, USG%, ORtg, DRtg, ...)
    team_standings.csv   - wins, losses, ratings, pace per team
    box_scores.csv       - per-player lines for every regular season game

Usage:
    python nba_scraper.py                          # full run
    python nba_scraper.py --skip-boxscores         # skip box scores
    python nba_scraper.py --output-dir ./data      # custom output dir
    python nba_scraper.py --test                   # connection test only
"""

import argparse
import logging
import os
import sys
import time
import random
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from nba_api.stats.endpoints import (
    commonteamyears,
    leaguedashplayerstats,
    leaguedashteamstats,
    leaguegamelog,
    leaguestandings,
)

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────
SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]

SLEEP_MIN = 2.0
SLEEP_MAX = 5.0

MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 3

DEFAULT_TIMEOUT = 180  # bulk endpoints return large payloads

REQUEST_HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Connection": "keep-alive",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "sec-ch-ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
}

# ─────────────────────────────────────────────────────────────────────
# Logging (UTF-8 safe for Windows)
# ─────────────────────────────────────────────────────────────────────
_console = logging.StreamHandler(
    open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        _console,
        logging.FileHandler("nba_scraper.log", mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger("nba_scraper")


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
_session = None

def get_session():
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update(REQUEST_HEADERS)
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=1, pool_maxsize=5,
            max_retries=requests.packages.urllib3.util.retry.Retry(
                total=2, backoff_factor=1,
                status_forcelist=[502, 503, 504],
                allowed_methods=["GET"],
            ),
        )
        _session.mount("https://", adapter)
    return _session

def reset_session():
    global _session
    if _session:
        try:
            _session.close()
        except Exception:
            pass
    _session = None
    log.info("Session reset")

def rate_limit_sleep():
    time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

def api_call_with_retry(endpoint_cls, max_retries=MAX_RETRIES, **kwargs):
    kwargs.setdefault("headers", REQUEST_HEADERS)
    kwargs.setdefault("timeout", DEFAULT_TIMEOUT)
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            rate_limit_sleep()
            return endpoint_cls(**kwargs, get_request=True)
        except Exception as exc:
            last_exc = exc
            base_wait = RETRY_BACKOFF_BASE ** attempt
            wait = min(base_wait * random.uniform(0.7, 1.3), 300)
            log.warning(
                "Attempt %d/%d for %s failed: %s — retrying in %.0fs",
                attempt, max_retries, endpoint_cls.__name__, exc, wait,
            )
            time.sleep(wait)
            if attempt >= 2 and attempt % 2 == 0:
                reset_session()
    raise RuntimeError(
        f"All {max_retries} attempts failed for {endpoint_cls.__name__}: {last_exc}"
    )


# ─────────────────────────────────────────────────────────────────────
# 1. Player Per-Game Stats (5 API calls — one per season)
# ─────────────────────────────────────────────────────────────────────
def fetch_player_pergame(seasons):
    frames = []
    for season in seasons:
        log.info("Fetching player per-game stats for %s ...", season)
        try:
            resp = api_call_with_retry(
                leaguedashplayerstats.LeagueDashPlayerStats,
                season=season,
                per_mode_detailed="PerGame",
                measure_type_detailed_defense="Base",
                season_type_all_star="Regular Season",
            )
            df = resp.get_data_frames()[0]
            df["SEASON"] = season
            frames.append(df)
            log.info("  -> %d players", len(df))
        except Exception as exc:
            log.error("FAILED player per-game for %s: %s", season, exc)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    col_map = {
        "PLAYER_ID": "PLAYER_ID", "PLAYER_NAME": "PLAYER_NAME",
        "TEAM_ABBREVIATION": "TEAM", "GP": "GP", "MIN": "MPG",
        "PTS": "PPG", "REB": "RPG", "AST": "APG", "STL": "SPG",
        "BLK": "BPG", "TOV": "TOV", "FG_PCT": "FG_PCT",
        "FG3_PCT": "3P_PCT", "FT_PCT": "FT_PCT", "SEASON": "SEASON",
    }
    out = combined.rename(columns=col_map)
    return out[[c for c in col_map.values() if c in out.columns]]


# ─────────────────────────────────────────────────────────────────────
# 2. Player Advanced Stats (5 API calls)
# ─────────────────────────────────────────────────────────────────────
def fetch_player_advanced(seasons):
    frames = []
    for season in seasons:
        log.info("Fetching player advanced stats for %s ...", season)
        try:
            resp = api_call_with_retry(
                leaguedashplayerstats.LeagueDashPlayerStats,
                season=season,
                per_mode_detailed="PerGame",
                measure_type_detailed_defense="Advanced",
                season_type_all_star="Regular Season",
            )
            df = resp.get_data_frames()[0]
            df["SEASON"] = season
            frames.append(df)
            log.info("  -> %d players", len(df))
        except Exception as exc:
            log.error("FAILED player advanced for %s: %s", season, exc)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    col_map = {
        "PLAYER_ID": "PLAYER_ID", "PLAYER_NAME": "PLAYER_NAME",
        "TEAM_ABBREVIATION": "TEAM", "GP": "GP",
        "TS_PCT": "TS_PCT", "USG_PCT": "USG_PCT",
        "OFF_RATING": "ORTG", "DEF_RATING": "DRTG",
        "NET_RATING": "NET_RATING", "PACE": "PACE",
        "PIE": "PIE", "SEASON": "SEASON",
    }
    out = combined.rename(columns=col_map)
    keep = [c for c in col_map.values() if c in out.columns]
    out = out[keep].copy()
    for col in ["PER", "BPM", "VORP", "WS", "WS_PER_48"]:
        if col not in out.columns:
            out[col] = float("nan")
    return out


# ─────────────────────────────────────────────────────────────────────
# 3. Team Standings (10 API calls — 2 per season)
# ─────────────────────────────────────────────────────────────────────
def fetch_team_standings(seasons):
    frames = []
    for season in seasons:
        log.info("Fetching team standings for %s ...", season)
        try:
            stand_resp = api_call_with_retry(
                leaguestandings.LeagueStandings,
                season=season, season_type="Regular Season",
            )
            stand_df = stand_resp.get_data_frames()[0]
        except Exception as exc:
            log.error("FAILED standings for %s: %s", season, exc)
            continue

        try:
            adv_resp = api_call_with_retry(
                leaguedashteamstats.LeagueDashTeamStats,
                season=season, per_mode_detailed="PerGame",
                measure_type_detailed_defense="Advanced",
                season_type_all_star="Regular Season",
            )
            adv_df = adv_resp.get_data_frames()[0]
        except Exception as exc:
            log.warning("No advanced team stats for %s: %s", season, exc)
            adv_df = pd.DataFrame()

        merged = stand_df.copy()
        if "TeamID" in merged.columns and "TEAM_ID" not in merged.columns:
            merged = merged.rename(columns={"TeamID": "TEAM_ID"})
        if not adv_df.empty:
            adv_cols = ["TEAM_ID"] + [
                c for c in ["OFF_RATING", "DEF_RATING", "NET_RATING", "PACE"]
                if c in adv_df.columns
            ]
            merged = merged.merge(adv_df[adv_cols], on="TEAM_ID", how="left")

        merged["SEASON"] = season
        frames.append(merged)
        log.info("  -> %d teams", len(merged))

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    col_map = {
        "TEAM_ID": "TEAM_ID", "TeamCity": "TEAM_CITY",
        "TeamName": "TEAM_NAME", "TEAM_NAME": "TEAM_NAME",
        "Conference": "CONFERENCE", "PlayoffRank": "CONF_RANK",
        "WINS": "W", "LOSSES": "L", "WinPCT": "WIN_PCT",
        "OFF_RATING": "OFF_RATING", "DEF_RATING": "DEF_RATING",
        "NET_RATING": "NET_RATING", "PACE": "PACE", "SEASON": "SEASON",
    }
    out = combined.rename(columns=col_map)
    seen = set()
    keep = []
    for c in col_map.values():
        if c in out.columns and c not in seen:
            keep.append(c)
            seen.add(c)
    out = out[keep].copy()
    if "SOS" not in out.columns:
        out["SOS"] = float("nan")
    return out


# ─────────────────────────────────────────────────────────────────────
# 4. Box Scores via LeagueGameLog (10 API calls total!)
#
#    Instead of hitting BoxScoreTraditionalV3 for each of ~6,150 games
#    (12,000+ API calls), LeagueGameLog returns ALL player game logs
#    for an entire season in ONE call.
#
#    player_or_team='P' -> every player's line for every game (~40k rows)
#    player_or_team='T' -> every team's line for every game (~2,460 rows)
#
#    We merge them to get per-player stats + game-level scores.
#
#    Trade-off: we lose quarter-by-quarter scores (Q1/Q2/Q3/Q4),
#    but we get reliable data that actually finishes.
# ─────────────────────────────────────────────────────────────────────
def fetch_box_scores(seasons):
    all_frames = []

    for season in seasons:
        log.info("=" * 55)
        log.info("BOX SCORES: %s (2 API calls for entire season)", season)
        log.info("=" * 55)

        # ── Player game logs (1 call) ─────────────────────────
        log.info("  Fetching player game logs for %s ...", season)
        try:
            player_resp = api_call_with_retry(
                leaguegamelog.LeagueGameLog,
                season=season,
                player_or_team_abbreviation="P",
                season_type_all_star="Regular Season",
            )
            player_df = player_resp.get_data_frames()[0]
            log.info("  -> %d player game log rows", len(player_df))
        except Exception as exc:
            log.error("FAILED player game logs for %s: %s", season, exc)
            continue

        # ── Team game logs (1 call) ───────────────────────────
        log.info("  Fetching team game logs for %s ...", season)
        try:
            team_resp = api_call_with_retry(
                leaguegamelog.LeagueGameLog,
                season=season,
                player_or_team_abbreviation="T",
                season_type_all_star="Regular Season",
            )
            team_df = team_resp.get_data_frames()[0]
            log.info("  -> %d team game log rows", len(team_df))
        except Exception as exc:
            log.error("FAILED team game logs for %s: %s", season, exc)
            # Still have player data — continue without game scores
            player_df["SEASON"] = season
            all_frames.append(player_df)
            continue

        # ── Build game-level info from team logs ──────────────
        # Each game has 2 rows in team_df (one per team).
        # MATCHUP contains "LAL vs. GSW" (home) or "LAL @ GSW" (away)
        game_info = _build_game_info(team_df)

        # ── Merge player logs with game info ──────────────────
        player_df["SEASON"] = season
        player_df = player_df.merge(game_info, on="GAME_ID", how="left")
        all_frames.append(player_df)

        log.info("  -> Season %s complete!", season)

        # Brief pause between seasons
        time.sleep(10)

    if not all_frames:
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)

    # ── Rename to final schema ────────────────────────────────
    rename = {
        "GAME_ID": "GAME_ID",
        "GAME_DATE": "GAME_DATE",
        "PLAYER_ID": "PLAYER_ID",
        "PLAYER_NAME": "PLAYER_NAME",
        "TEAM_ABBREVIATION": "TEAM",
        "MIN": "MIN",
        "PTS": "PTS",
        "REB": "REB",
        "AST": "AST",
        "STL": "STL",
        "BLK": "BLK",
        "TOV": "TOV",
        "FGA": "FGA",
        "FGM": "FGM",
        "FG3A": "3PA",
        "FG3M": "3PM",
        "FTA": "FTA",
        "FTM": "FTM",
        "PLUS_MINUS": "PLUS_MINUS",
        "HOME_TEAM": "HOME_TEAM",
        "AWAY_TEAM": "AWAY_TEAM",
        "HOME_PTS": "HOME_PTS",
        "AWAY_PTS": "AWAY_PTS",
        "SEASON": "SEASON",
    }
    combined = combined.rename(columns=rename)
    keep = [c for c in rename.values() if c in combined.columns]
    seen = set()
    keep_dedup = []
    for c in keep:
        if c not in seen:
            keep_dedup.append(c)
            seen.add(c)
    return combined[keep_dedup].copy()


def _build_game_info(team_df):
    """
    From team game logs, build a game-level lookup with
    home/away teams and final scores.

    MATCHUP format: "LAL vs. GSW" means LAL is home.
                    "LAL @ GSW" means LAL is away (GSW is home).
    """
    games = {}

    for _, row in team_df.iterrows():
        gid = row["GAME_ID"]
        team = row["TEAM_ABBREVIATION"]
        pts = row["PTS"]
        matchup = row.get("MATCHUP", "")

        if gid not in games:
            games[gid] = {"GAME_ID": gid, "GAME_DATE": row.get("GAME_DATE", "")}

        # "vs." = this team is home; "@" = this team is away
        if " vs. " in str(matchup):
            games[gid]["HOME_TEAM"] = team
            games[gid]["HOME_PTS"] = pts
        elif " @ " in str(matchup):
            games[gid]["AWAY_TEAM"] = team
            games[gid]["AWAY_PTS"] = pts

    return pd.DataFrame(list(games.values()))


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="NBA data scraper v2 (bulk endpoints)")
    parser.add_argument(
        "--output-dir", default=".", help="Directory for output CSVs",
    )
    parser.add_argument(
        "--skip-boxscores", action="store_true",
        help="Skip box score collection",
    )
    parser.add_argument(
        "--seasons", nargs="+", default=SEASONS,
        help="Seasons to scrape (default: 2021-22 through 2025-26)",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run connection test then exit",
    )
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Connection test ───────────────────────────────────────
    log.info("Testing connection to stats.nba.com ...")
    try:
        probe = api_call_with_retry(
            commonteamyears.CommonTeamYears, max_retries=3,
        )
        teams = probe.get_data_frames()[0]
        log.info("  OK — %d team-year records returned", len(teams))
    except Exception as exc:
        log.error("  Connection FAILED: %s", exc)
        log.error("  Check your network or try again in a few minutes.")
        sys.exit(1)

    if args.test:
        log.info("Test passed — exiting.")
        sys.exit(0)

    start_time = datetime.now()
    log.info("NBA Scraper v2 started at %s", start_time.strftime("%Y-%m-%d %H:%M:%S"))
    log.info("Seasons: %s", ", ".join(args.seasons))
    log.info("Output: %s", outdir.resolve())
    log.info("")

    # ── 1. Player Per-Game Stats ──────────────────────────────
    log.info("=" * 55)
    log.info("STEP 1/4: Player Per-Game Stats (5 API calls)")
    log.info("=" * 55)
    pergame_df = fetch_player_pergame(args.seasons)
    if not pergame_df.empty:
        path = outdir / "player_pergame.csv"
        pergame_df.to_csv(path, index=False)
        log.info("  SAVED %s (%d rows)", path.name, len(pergame_df))
    else:
        log.warning("No player per-game data collected.")
    time.sleep(10)

    # ── 2. Player Advanced Stats ──────────────────────────────
    log.info("")
    log.info("=" * 55)
    log.info("STEP 2/4: Player Advanced Stats (5 API calls)")
    log.info("=" * 55)
    advanced_df = fetch_player_advanced(args.seasons)
    if not advanced_df.empty:
        path = outdir / "player_advanced.csv"
        advanced_df.to_csv(path, index=False)
        log.info("  SAVED %s (%d rows)", path.name, len(advanced_df))
    else:
        log.warning("No player advanced data collected.")
    time.sleep(10)

    # ── 3. Team Standings ─────────────────────────────────────
    log.info("")
    log.info("=" * 55)
    log.info("STEP 3/4: Team Standings (10 API calls)")
    log.info("=" * 55)
    standings_df = fetch_team_standings(args.seasons)
    if not standings_df.empty:
        path = outdir / "team_standings.csv"
        standings_df.to_csv(path, index=False)
        log.info("  SAVED %s (%d rows)", path.name, len(standings_df))
    else:
        log.warning("No team standings data collected.")
    time.sleep(10)

    # ── 4. Box Scores ─────────────────────────────────────────
    if args.skip_boxscores:
        log.info("")
        log.info("=" * 55)
        log.info("STEP 4/4: Box Scores — SKIPPED")
        log.info("=" * 55)
    else:
        log.info("")
        log.info("=" * 55)
        log.info("STEP 4/4: Box Scores (10 API calls — bulk method)")
        log.info("=" * 55)
        boxscore_df = fetch_box_scores(args.seasons)
        if not boxscore_df.empty:
            path = outdir / "box_scores.csv"
            boxscore_df.to_csv(path, index=False)
            log.info("  SAVED %s (%d rows)", path.name, len(boxscore_df))
        else:
            log.warning("No box score data collected.")

    # ── Summary ───────────────────────────────────────────────
    elapsed = datetime.now() - start_time
    log.info("")
    log.info("=" * 55)
    log.info("COMPLETE — total time %s", str(elapsed).split(".")[0])
    log.info("=" * 55)
    log.info("Files in %s:", outdir.resolve())
    for f in sorted(outdir.glob("*.csv")):
        size_mb = f.stat().st_size / (1024 * 1024)
        log.info("  %s  (%.1f MB)", f.name, size_mb)


if __name__ == "__main__":
    main()
