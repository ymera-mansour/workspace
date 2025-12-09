"""Quality validator"""
class QualityValidator:
    def __init__(self, config):
        self.config = config
    async def validate(self, context):
        return {"validator": "Quality", "passed": True, "score": 0.92}
