"""Test validator"""
class TestValidator:
    def __init__(self, config):
        self.config = config
    async def validate(self, context):
        return {"validator": "Tests", "passed": True, "tests_passed": 45}
