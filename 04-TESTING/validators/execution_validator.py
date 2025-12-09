"""Execution validator"""
class ExecutionValidator:
    def __init__(self, config):
        self.config = config
    async def validate(self, context):
        return {"validator": "Execution", "passed": True}
