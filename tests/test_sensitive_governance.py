from __future__ import annotations

from personal_agent_memory.sensitive import inspect_sensitive_text


def test_secret_patterns_include_segmented_tokens() -> None:
    assert inspect_sensitive_text("api_key=abcdefghijklmno12345") is not None
    assert inspect_sensitive_text("sk-abcde fghij klmno pqrst uvwxyz") is not None
    assert inspect_sensitive_text("-----BEGIN PRIVATE KEY-----") is not None
    assert inspect_sensitive_text("an example password placeholder") is None
    assert inspect_sensitive_text("# Secret management\n\nRotate deployment credentials.") is None
    assert inspect_sensitive_text("Passwords are configured by deployment automation.") is None
    assert inspect_sensitive_text("password: generated-at-deploy-time") is None
    assert inspect_sensitive_text("password=${DATABASE_PASSWORD}") is None
    assert inspect_sensitive_text("api_key: managed-by-vault") is None
    assert inspect_sensitive_text("password: correct-horse-battery-staple") is not None
    assert inspect_sensitive_text('DB_PASSWORD="correct horse battery staple"') is not None
    assert inspect_sensitive_text("password='s3cr3t'") is not None
    assert inspect_sensitive_text("password='generated at deploy time'") is None
    assert inspect_sensitive_text('"password": "correct horse battery staple"') is not None
    assert inspect_sensitive_text("'client_secret' = 'real client secret'") is not None
    assert inspect_sensitive_text('"password": "placeholder actual-secret"') is not None
    assert inspect_sensitive_text("'token' = 'sample-abcdefghijklmno'") is not None
    assert inspect_sensitive_text('"password": "placeholder"') is None
    assert inspect_sensitive_text(
        'DB_PASSWORD="generated at deploy time"; CLIENT_SECRET="real client secret"'
    ) is not None
    for variable in ("DB_PASSWORD", "CLIENT_SECRET", "API_KEY", "ACCESS_TOKEN"):
        assert inspect_sensitive_text(f"{variable}=correct-horse-battery-staple") is not None
    assert inspect_sensitive_text("API_KEY=abc1234") is not None
    assert inspect_sensitive_text("ACCESS_TOKEN=tok1234") is not None
    assert inspect_sensitive_text("DB_PASS=s3cr3t") is not None
    assert inspect_sensitive_text(f"AWS_SECRET_ACCESS_KEY={'a' * 40}") is not None
    assert inspect_sensitive_text("DB_PASS=${DATABASE_PASSWORD}") is None
    assert inspect_sensitive_text("AWS_SECRET_ACCESS_KEY=managed-by-vault") is None
    assert inspect_sensitive_text("DB_PASS=vault://secret/data/app#password") is None
    assert inspect_sensitive_text("DB_PASS=vault://secret/data/app?token=real#password") is not None
    assert inspect_sensitive_text(
        "DB_PASSWORD=${DATABASE_PASSWORD}\nCLIENT_SECRET=correct-horse-battery-staple"
    ) is not None
    assert inspect_sensitive_text("The deployment credential may need rotation.") is not None
