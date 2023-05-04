from nyckel.greeter import say_hi_to_me


def test_simple():
    assert say_hi_to_me("oscar") == "hello oscar"