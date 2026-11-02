"""CORS policy: deployment defaults plus any explicitly configured patterns."""

import re

from app.core.config import PREVIEW_ORIGIN_REGEX, Settings


def make(**overrides) -> Settings:
    """Build settings without reading the developer's .env file."""
    return Settings(_env_file=None, **overrides)


def matches(settings: Settings, origin: str) -> bool:
    pattern = settings.effective_cors_origin_regex
    if pattern is None:
        return origin in settings.cors_origins
    return origin in settings.cors_origins or bool(re.fullmatch(pattern, origin))


def test_vercel_and_localhost_are_allowed_by_default():
    settings = make()
    assert matches(settings, "https://sany-portfolio-one.vercel.app")
    assert matches(settings, "https://my-portfolio-git-main-sany.vercel.app")
    assert matches(settings, "http://localhost:3005")


def test_unrelated_origins_are_rejected():
    settings = make()
    assert not matches(settings, "https://evil.example.com")
    assert not matches(settings, "https://vercel.app.evil.example.com")


def test_explicit_regex_adds_to_the_defaults():
    """A stale dashboard value must not lock out the deployed frontend."""
    settings = make(
        cors_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
        cors_origins_raw="",
    )

    assert matches(settings, "http://localhost:3005")
    assert matches(settings, "https://sany-portfolio-one.vercel.app")


def test_custom_domain_pattern_is_added():
    settings = make(
        cors_origin_regex=r"^https://sany\.dev$", cors_origins_raw=""
    )
    assert matches(settings, "https://sany.dev")
    assert matches(settings, "https://sany-portfolio-one.vercel.app")


def test_previews_can_be_disabled():
    settings = make(
        cors_allow_previews=False,
        cors_origin_regex=r"^https://sany\.dev$",
        cors_origins_raw="",
    )
    assert matches(settings, "https://sany.dev")
    assert not matches(settings, "https://sany-portfolio-one.vercel.app")


def test_pattern_is_anchored():
    assert PREVIEW_ORIGIN_REGEX.startswith("^")
    settings = make(cors_origins_raw="")
    assert not matches(settings, "http://localhost:3000.evil.com")
