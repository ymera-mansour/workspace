"""Expert validator"""
class ExpertValidator:
    def __init__(self, config):
        self.config = config
    async def validate(self, context):
        return {"validator": "Expert", "passed": True, "approved": True}
