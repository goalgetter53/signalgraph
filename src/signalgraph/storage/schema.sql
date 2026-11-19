-- signalgraph DuckDB schema. All timestamps are naive UTC.

CREATE TABLE IF NOT EXISTS signals (
    event_id        VARCHAR PRIMARY KEY,      -- sha256(source, native_id)[:20]
    source          VARCHAR NOT NULL,
    native_id       VARCHAR NOT NULL,
    signal_type     VARCHAR NOT NULL,
    subtype         VARCHAR NOT NULL,
    company_name    VARCHAR NOT NULL,
    company_domain  VARCHAR,
    location        VARCHAR,
    country         VARCHAR,
    occurred_at     TIMESTAMP NOT NULL,
    title           VARCHAR NOT NULL,
    url             VARCHAR,
    magnitude       DOUBLE NOT NULL,
    attributes      JSON,
    mention_id      VARCHAR NOT NULL,
    first_run_id    VARCHAR NOT NULL,
    last_seen_at    TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS mentions (
    mention_id  VARCHAR PRIMARY KEY,
    source      VARCHAR NOT NULL,
    raw_name    VARCHAR NOT NULL,
    raw_domain  VARCHAR,
    location    VARCHAR,
    country     VARCHAR,
    aliases     JSON
);

CREATE TABLE IF NOT EXISTS accounts (
    account_id    VARCHAR PRIMARY KEY,
    name          VARCHAR NOT NULL,
    domain        VARCHAR,
    country       VARCHAR,
    aliases       JSON,
    member_count  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS account_members (
    mention_id  VARCHAR PRIMARY KEY,
    account_id  VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS scores (
    account_id   VARCHAR PRIMARY KEY,
    score        DOUBLE NOT NULL,
    raw_points   DOUBLE NOT NULL,
    by_type      JSON,
    icp_fit      BOOLEAN NOT NULL,
    icp_reasons  JSON,
    breakdown    JSON,
    as_of        TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS watermarks (
    connector_key  VARCHAR PRIMARY KEY,
    watermark      TIMESTAMP NOT NULL,
    updated_at     TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id       VARCHAR PRIMARY KEY,
    kind         VARCHAR NOT NULL,
    started_at   TIMESTAMP NOT NULL,
    finished_at  TIMESTAMP,
    stats        JSON
);
