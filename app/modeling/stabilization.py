from __future__ import annotations

# Placeholder stabilization priors (games). Adjust when historical data is available.
STABILIZATION_GAMES: dict[str, float] = {
    "Points": 10.0,
    "Rebounds": 12.0,
    "Assists": 12.0,
    "Steals": 20.0,
    "Blocked Shots": 20.0,
    "Turnovers": 12.0,
    "3-PT Made": 18.0,
    "3-PT Attempted": 18.0,
    "FG Made": 10.0,
    "FG Attempted": 10.0,
    "Free Throws Made": 8.0,
    "Free Throws Attempted": 8.0,
    "Pts+Rebs": 10.0,
    "Pts+Asts": 10.0,
    "Rebs+Asts": 10.0,
    "Pts+Rebs+Asts": 10.0,
    "Blks+Stls": 20.0,
    "Two Pointers Made": 12.0,
    "Two Pointers Attempted": 12.0,
}
