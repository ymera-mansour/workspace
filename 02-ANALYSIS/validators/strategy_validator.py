"""Strategy validator"""
class StrategyValidator:
    def __init__(self, config):
        self.config = config
    async def validate(self, context):
        return {"validator": "Strategy", "passed": True}
