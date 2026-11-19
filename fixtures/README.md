# Recorded fixtures (offline tests and `make demo`)

Every company in this directory is **fictional** (Northwind Robotics, Zentro,
Kestrel Data, ...). The files are written in the exact response shape of the
real public APIs (shapes checked against live responses on 2026-11-19 for
EDGAR full-text search, Form D `primary_doc.xml`, Greenhouse, Lever and Ashby;
GitHub org events and RSS/Atom follow the documented formats).

`manifest.yaml` maps `host + path` to a file. `FixtureTransport` serves them
and returns 404 for anything else, so a test can never reach the network.
