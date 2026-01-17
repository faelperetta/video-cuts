import pytest
from videocuts.config import Config, ProfanityConfig
from videocuts.caption.generators import apply_profanity_filter

def test_profanity_filter_disabled():
    cfg = Config()
    cfg.profanity.enabled = False
    text = "This is a fuck bad word."
    assert apply_profanity_filter(text, cfg) == text

def test_profanity_filter_enabled_default():
    cfg = Config()
    cfg.profanity.enabled = True
    # "fuck" should be replaced by "f***"
    text = "This is a fuck bad word."
    expected = "This is a f*** bad word."
    assert apply_profanity_filter(text, cfg) == expected

def test_profanity_filter_case_sensitivity():
    cfg = Config()
    cfg.profanity.enabled = True
    text = "FUCK this."
    # better-profanity is usually case-insensitive by default
    expected = "F*** this."
    assert apply_profanity_filter(text, cfg) == expected

def test_profanity_filter_punctuation():
    cfg = Config()
    cfg.profanity.enabled = True
    text = "What the fuck!"
    expected = "What the f***!"
    assert apply_profanity_filter(text, cfg) == expected

def test_profanity_filter_custom_words():
    cfg = Config()
    cfg.profanity.enabled = True
    cfg.profanity.custom_words = ["apple", "banana"]
    
    text = "I like apple and banana."
    expected = "I like a**** and b*****."
    assert apply_profanity_filter(text, cfg) == expected

def test_profanity_filter_short_word():
    cfg = Config()
    cfg.profanity.enabled = True
    cfg.profanity.custom_words = ["badword"]
    text = "Just a badword."
    # "badword" -> "b******"
    expected = "Just a b******."
    assert apply_profanity_filter(text, cfg) == expected
