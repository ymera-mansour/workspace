"""Leaderboard Generator - Rankings and performance metrics"""
class LeaderboardGenerator:
    def __init__(self):
        self.entries = []
    def generate(self):
        return sorted(self.entries, key=lambda x: x.get("score", 0), reverse=True)
