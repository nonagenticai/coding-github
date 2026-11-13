"""Identity headers for calls coding-github makes to other services.

Two kinds of outbound call leave this service and they must never be confused:

1. **GitHub.com** — ``github_app_auth``, ``github_actions_trigger``,
   ``github/github_service``. Those carry a GitHub App JWT / installation token
   / user PAT in ``Authorization``. Nothing in this module touches them, and
   nothing in this module may ever be applied to them: overwriting a PAT with
   our own licence would break every repo operation, and sending a GitHub PAT
   to a sibling would leak a customer's credential across a service boundary.

2. **Siblings inside the mesh** — coding-containers, coding-analysis,
   coding-security. Every one of those calls currently sends a *bare*
   ``X-Tenant-Id`` and is accepted only because the callee still trusts that
   header from the pod CIDR. That is a network position, not a credential. When
   a callee runs with ``TRUST_TENANT_HEADER=false`` the bare header is refused,
   and in this service a refusal is invisible: the provisioning scans are
   fire-and-forget with the failure logged and swallowed, and
   ``_analyze_security_vulnerabilities`` returns ``[]`` on any exception — a
   push event would report "no vulnerabilities" for a repo nobody scanned.

This module owns kind (2) only.

What goes on the wire for a sibling call
----------------------------------------
* ``Authorization: Bearer <CODING_SERVICE_TOKEN>`` — **this service's own**
  licence, never a forwarded copy of an inbound one and never a GitHub token.
  Most of these calls are detached (``asyncio.create_task`` scans, webhook
  handlers) and have no inbound request to borrow a credential from.
* ``X-Tenant-Id`` — the **customer** the call is made for.

They stay in separate headers on purpose. Folding the licence into
``X-Tenant-Id`` makes the callee resolve the tenant from our token and collapses
every customer into ``svc-coding-github``'s own tenant. The callee honours the
asserted tenant only when our subject is in its ``CODING_ONBEHALF_SUBJECTS``
allowlist.
"""

from __future__ import annotations

import os

from coding_client.auth.outbound import optional_tenant_headers

__all__ = ["optional_sibling_headers", "secrets_service_headers", "service_token"]


def service_token() -> str | None:
    """This service's own licence, read per call so a rotation needs no restart.

    Read from the environment at call time rather than captured at import: a
    module constant would freeze whatever the value was when the first import
    ran. The ``or None`` normalises a set-but-empty variable to absent so a
    malformed bare ``Bearer `` never goes out.
    """
    return os.getenv("CODING_SERVICE_TOKEN") or None


def optional_sibling_headers(tenant_id: str | None) -> dict[str, str]:
    """Headers for a sibling call, attaching the tenant only when there is one.

    **Absent stays absent.** The header is omitted rather than filled with a
    default, a user id, or the literal ``"anonymous"``. Substituting anything
    makes the call claim a tenant no registry issued, which no audit trail can
    attribute and which is indistinguishable downstream from a real tenant that
    happens to be named that — while making a call that should have been refused
    succeed against someone else's partition.

    The credential is attached either way. Proving who we are is not a claim
    about whose data we want, and an unscoped call that can still name itself is
    strictly better than an anonymous one.
    """
    return optional_tenant_headers(tenant_id, service_token=service_token())


def secrets_service_headers(tenant_id: str | None) -> dict[str, str]:
    """Tenant header for the **secrets-service**, which is not a coding sibling.

    Deliberately carries no ``Authorization``: secrets-service is outside the
    coding-* licence fleet, it does not introspect a coding service licence, and
    ``CODING_ONBEHALF_SUBJECTS`` means nothing to it. Sending our licence there
    would put a fleet credential in front of a service that has no way to
    validate it — and no reason to see it.

    What it *does* share with the sibling path is the rule above: a missing
    tenant is omitted, never invented, so an unattributable secret fetch is
    refused by secrets-service instead of being served out of a partition named
    after a fallback.
    """
    return optional_tenant_headers(tenant_id)
