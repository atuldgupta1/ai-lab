from unittest.mock import patch
from anthropic import APIConnectionError
from src.llm.client import complete

# Make the API call raise a connection error the first 2 times, then succeed
call_count = {"n": 0}
real_create = None

with patch("src.llm.client.Anthropic") as MockClient:
    def flaky_create(**kwargs):
        call_count["n"] += 1
        print(f"  -> attempt {call_count['n']}")
        if call_count["n"] < 3:
            raise APIConnectionError(request=None)
        # Fake a minimal successful response object
        class FakeUsage: input_tokens = 5; output_tokens = 5
        class FakeBlock: text = "hello there friend"
        class FakeResponse:
            usage = FakeUsage()
            content = [FakeBlock()]
        return FakeResponse()

    MockClient.return_value.messages.create = flaky_create
    print(complete("hi"))