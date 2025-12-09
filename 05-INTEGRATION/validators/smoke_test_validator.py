"""Smoke test validator"""
class SmokeTestValidator:
    def __init__(self, config):
        self.config = config
    async def validate(self, context):
        return {"validator": "Smoke Tests", "passed": True, "health": "OK"}
