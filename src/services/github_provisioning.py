"""
GitHub Provisioning Service - Main orchestration service for GitHub container provisioning.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any

import httpx
from coding_client.auth.tenant import DEFAULT_TENANT_ID, TENANT_ID_HEADER
from coding_client.exceptions.api_exceptions import NotFoundException
from coding_client.models.container import ContainerInfo, GitHubRepoRequest

from config.settings import get_settings
from services.github_validation import (
    CoreGitHubValidationService as GitHubValidationService,
)
from services.outbound import optional_sibling_headers, secrets_service_headers
from services.tenant_binding import TenantRepoTokenError

logger = logging.getLogger(__name__)

# Reserved for single-purpose workspaces owned by their creator (fr quality:
# ``q-`` + 12 lowercase hex). fr deletes and re-clones these freely, so they are
# never handed out by repo-url reuse; only an exact-name lookup may return one.
DEDICATED_WORKSPACE_RE = re.compile(r"q-[0-9a-f]{12}")


class WorkspaceRepoMismatchError(TenantRepoTokenError):
    """An exact-name workspace exists but is not attributable to the requested repo.

    Raised when ``exact_name`` names a workspace whose ``repo_url`` differs from the
    request's (or is absent). Returning it would hand back another repo's checkout
    carrying THIS repo's commit_sha; creating a second one under the same name is
    not possible either. So the request is refused, never "close enough".

    Subclasses :class:`TenantRepoTokenError` deliberately: ``provision_repository``
    already re-raises that family past its ``RuntimeError``/``Exception`` result
    envelopes, and the provision route maps it through
    ``tenant_error_http_exception`` into ``{status, error_code, details}`` — so the
    409 and its machine code reach the caller with no new handler.
    """

    code = "workspace_repo_mismatch"
    http_status = 409


class WorkspaceToolchainMismatchError(TenantRepoTokenError):
    """The workspace provision would reuse was built with a different toolchain.

    Raised only when the caller set ``toolchain`` explicitly and did not ask for
    ``force_rebuild``. Handing back a ``python`` workspace to a ``python+rust``
    request (or the reverse) would silently give the caller the wrong tools, and
    rebuilding it unasked would destroy whatever is in it. So: refuse, and name
    ``force_rebuild`` as the way through. Same 409 envelope as
    :class:`WorkspaceRepoMismatchError`.
    """

    code = "workspace_toolchain_mismatch"
    http_status = 409


class _WorkspaceInfo(ContainerInfo):
    """coding-client's ``ContainerInfo`` plus the workspace ``toolchain``.

    coding-containers#184 returns ``toolchain`` on GET/list workspace responses,
    but the shared ``ContainerInfo`` ignores unknown fields, so converting the raw
    dict dropped it before the reuse check could read it. Serialised as a plain
    ``ContainerInfo`` wherever a response model declares that type, so no outgoing
    schema changes.
    """

    toolchain: str | None = None


#: coding-containers' own rule: a workspace with no recorded toolchain is python.
_DEFAULT_TOOLCHAIN = "python"


def _owner_repo(repo_url: str) -> tuple[str | None, str | None]:
    """``https://github.com/o/r(.git)`` → ``("o", "r")``; anything shorter → Nones."""
    parts = _ContainersHTTPAdapter._normalize_repo_url(repo_url).split("/")
    if len(parts) >= 2 and parts[-2] and parts[-1]:
        return parts[-2], parts[-1]
    return None, None


def _scan_headers(tenant_id: str | None) -> dict[str, str]:
    """Outbound headers for the fire-and-forget provisioning scans.

    Sends ``X-Tenant-Id`` whenever a tenant is actually known: the caller's, or
    an explicitly configured ``DEFAULT_TENANT_ID``. When the tenant is unknown
    these calls used to go out with *no* headers at all, which only worked
    because coding-analysis and coding-security fell back to trusting any
    RFC1918 source IP — a network position, not a credential, covering the whole
    pod CIDR.

    ⚠️ ``DEFAULT_TENANT_ID`` is ``str | None`` as of coding-client 1.9.0 (#36),
    which removed the built-in ``"n8n-dev"`` fallback: unset now means unset
    rather than a stale pre-uuid7 label. This service does not set the variable
    in its Deployment, so unset is the live case, and ``tenant_id or
    DEFAULT_TENANT_ID`` would have put a literal ``None`` into a header value.
    Inventing a tenant here is exactly what #36 exists to stop, so the header is
    omitted instead and the scan is refused by the callee — logged and swallowed
    like any other scan failure, rather than silently attributed to whatever
    tenant a fallback happened to name.

    The header *building* now goes through the SDK (``optional_tenant_headers``
    via ``optional_sibling_headers``) rather than a hand-rolled dict, because
    that is where the omit-don't-default rule and the ``Authorization`` split
    are tested once for the whole fleet. What could not be delegated — and is
    why this function still exists — is the ``DEFAULT_TENANT_ID`` resolution and
    the warning above: the SDK cannot know that an absent tenant here is a
    deployment gap worth logging rather than an ordinary unscoped call.

    The scan also now carries this service's own licence, so it survives a
    callee running with ``TRUST_TENANT_HEADER=false``. The credential goes out
    even when no tenant is resolved: naming ourselves is not a claim about whose
    data we want, and it makes the callee's refusal attributable to
    ``svc-coding-github`` in its logs instead of anonymous.
    """
    resolved = tenant_id or DEFAULT_TENANT_ID
    if resolved is None:
        logger.warning(
            "No tenant for provisioning scan and DEFAULT_TENANT_ID is unset; "
            "sending no %s. The scan will be rejected by the callee. Set "
            "DEFAULT_TENANT_ID on the Deployment if these scans should run.",
            TENANT_ID_HEADER,
        )
    return optional_sibling_headers(resolved)


async def _trigger_full_scan(container_name: str, tenant_id: str | None) -> None:
    """Fire-and-forget full AST scan against coding-analysis after container creation.

    Uses a short request timeout (only the scan kickoff ack, not the scan duration —
    coding-analysis runs the scan in the background). Failures are logged and swallowed
    so they never surface to the provision response.
    """
    analysis_url = get_settings().coding_analysis_url
    headers = _scan_headers(tenant_id)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{analysis_url}/api/analysis/scan/scan",
                json={"container_name": container_name, "action": "full_scan"},
                headers=headers,
            )
        if response.status_code >= 400:
            logger.warning(
                "Auto-index scan kickoff for %s returned HTTP %s: %s",
                container_name,
                response.status_code,
                response.text[:500],
            )
        else:
            logger.info("Auto-index scan kickoff accepted for %s", container_name)
    except httpx.HTTPError as exc:
        logger.warning("Auto-index scan kickoff failed for %s: %s", container_name, exc)


async def _trigger_dependency_scan(container_name: str, tenant_id: str | None) -> None:
    """Fire-and-forget dependency scan against coding-security after provision.

    Same semantics as ``_trigger_full_scan``: short request timeout (we're
    waiting on the kickoff ack, not the scan duration), always a tenant header,
    errors logged and swallowed so the provision response is never held up.
    """
    security_url = get_settings().coding_security_url
    headers = _scan_headers(tenant_id)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{security_url}/api/dependencies/{container_name}/scan",
                headers=headers,
            )
        if response.status_code >= 400:
            logger.warning(
                "Auto dependency-scan kickoff for %s returned HTTP %s: %s",
                container_name,
                response.status_code,
                response.text[:500],
            )
        else:
            logger.info("Auto dependency-scan kickoff accepted for %s", container_name)
    except httpx.HTTPError as exc:
        logger.warning("Auto dependency-scan kickoff failed for %s: %s", container_name, exc)


class _ContainersHTTPAdapter:
    """Thin adapter that exposes docker_service-like methods via the containers HTTP client."""

    @staticmethod
    def _unwrap_container(data: Any) -> Any:
        # coding-containers wraps GET /api/workspaces/{name} in a status envelope:
        # {status, message, timestamp, request_id, container: {...}}. Strip it.
        if isinstance(data, dict) and isinstance(data.get("container"), dict):
            return data["container"]
        return data

    @staticmethod
    def _unwrap_containers(data: Any) -> list:
        # coding-containers wraps GET /api/workspaces/ in a status envelope:
        # {status, message, ..., containers: [...]}.
        if isinstance(data, dict) and "containers" in data:
            return data.get("containers") or []
        return data if isinstance(data, list) else []

    @staticmethod
    def _normalize_repo_url(repo_url: Any) -> str:
        # Match repos regardless of a trailing slash or a ``.git`` suffix so
        # ``…/repo``, ``…/repo/`` and ``…/repo.git`` all dedupe to one container.
        return str(repo_url or "").strip().rstrip("/").removesuffix(".git").lower()

    @staticmethod
    def _container_field(container: Any, field: str) -> Any:
        if isinstance(container, dict):
            return container.get(field)
        return getattr(container, field, None)

    async def find_existing_container(self, repo_url: str, tenant_id: str | None = None) -> Any:
        from coding_client.clients import ServiceError

        from main import get_workspaces_client

        client = get_workspaces_client()
        # coding-containers scopes GET /api/workspaces/ by the ``X-Tenant-Id``
        # header (NOT a query param) and ignores an unknown ``repo_url`` param.
        # The previous code passed ``params={"repo_url": …}`` (silently dropped)
        # and returned ``containers[0]`` unconditionally, so provision handed back
        # an UNRELATED/stale container for the tenant's first deployment and never
        # created one for the requested repo. Forward the tenant and match the
        # repo client-side; also re-check tenant_id in case the inter-service
        # caller resolves to an admin identity (which lists every tenant).
        headers = optional_sibling_headers(tenant_id) or None
        target = self._normalize_repo_url(repo_url)
        try:
            data = await client.list_workspaces(headers=headers)
            for container in self._unwrap_containers(data):
                if self._normalize_repo_url(self._container_field(container, "repo_url")) != target:
                    continue
                name = self._container_field(container, "container_name")
                if isinstance(name, str) and DEDICATED_WORKSPACE_RE.fullmatch(name):
                    continue
                if tenant_id is not None and not self._container_belongs_to_tenant(
                    container, container, tenant_id
                ):
                    continue
                return _WorkspaceInfo(**container) if isinstance(container, dict) else container
        except ServiceError:
            pass
        return None

    async def find_container_by_name(self, name: str, tenant_id: str | None = None) -> Any:
        from coding_client.clients import ServiceError

        from main import get_workspaces_client

        client = get_workspaces_client()
        # Scope the lookup to the caller's tenant. coding-containers reads the
        # ``X-Tenant-Id`` header; we ALSO re-verify the returned container's
        # tenant_id client-side (mirroring ``find_existing_container``) in case
        # the inter-service caller resolves to an admin identity that can read
        # any tenant. Without this a caller could read another tenant's
        # container status/repo_url just by guessing its (repo-name-derived)
        # container name.
        headers = optional_sibling_headers(tenant_id) or None
        try:
            raw = await client.get_workspace(name, headers=headers)
        except ServiceError as e:
            if e.status_code == 404:
                return None
            raise
        data = self._unwrap_container(raw)
        if tenant_id is not None and not self._container_belongs_to_tenant(raw, data, tenant_id):
            return None
        return _WorkspaceInfo(**data) if isinstance(data, dict) else data

    @staticmethod
    def _container_belongs_to_tenant(raw: Any, container: Any, tenant_id: str) -> bool:
        """Return True only if the container is *positively* owned by ``tenant_id``.

        FAIL CLOSED. coding-containers reports the owning tenant either at the
        response envelope level (``{tenant_id, container: {...}}``) or on the
        container object itself; check both.

        An unlabelled container (``tenant_id`` missing / ``None`` / ``""``)
        belongs to NOBODY and matches no tenant. The previous
        ``c_tenant in (None, "", tenant_id)`` treated "no label" as "owned by
        whoever asks", which left the exact cross-tenant read/delete hole this
        ownership check exists to close: any tenant could read or delete an
        unlabelled container just by guessing its (repo-name-derived) name.

        Unlabelled containers are a real upstream state, not just legacy data:
        coding-containers' registry-MISS path rebuilds a ContainerInfo from the
        live Deployment and *deliberately* leaves ``tenant_id`` unset
        (``reconstruct_container_info``). Denying is still the right answer for
        a tenant-scoped request — ownership is unverifiable, so the caller gets
        a 404 and retries once the registry entry is back — but we log it so a
        registry blip is diagnosable instead of a silent disappearance.

        Callers that legitimately need an unscoped view (admin / internal, no
        tenant on the request) must pass ``tenant_id=None`` and skip this check
        entirely; no tenant-scoped route can reach that path (``require_tenant``
        either yields a non-empty tenant or raises 401).
        """
        c_tenant = None
        if isinstance(raw, dict):
            c_tenant = raw.get("tenant_id")
        if c_tenant in (None, ""):
            c_tenant = _ContainersHTTPAdapter._container_field(container, "tenant_id")
        if c_tenant in (None, ""):
            logger.warning(
                "Denying tenant-scoped access to unlabelled container %s (requesting tenant %s): "
                "ownership cannot be verified",
                _ContainersHTTPAdapter._container_field(container, "container_name"),
                tenant_id,
            )
            return False
        if tenant_id in (None, ""):
            return False
        return str(c_tenant) == str(tenant_id)

    async def remove_container(self, container: Any, tenant_id: str | None = None) -> bool:
        from main import get_workspaces_client

        client = get_workspaces_client()
        name = container.container_name if hasattr(container, "container_name") else str(container)
        # Forward the caller's tenant so coding-containers applies its own
        # ownership gate too. The container object reaching here has already
        # passed a fail-closed client-side ownership check (it came from
        # ``find_existing_container`` / ``list_containers``), but this delete
        # must never go out unscoped when a tenant is known.
        headers = optional_sibling_headers(tenant_id) or None
        await client.delete_workspace(name, headers=headers)
        return True

    async def remove_container_by_name(self, name: str, tenant_id: str | None = None) -> bool:
        from coding_client.clients import ServiceError

        from main import get_workspaces_client

        client = get_workspaces_client()
        # Ownership guard: only delete a container that belongs to this tenant.
        # ``find_container_by_name`` is tenant-scoped and returns None for
        # another tenant's (or a missing) container, so a tenant force-rebuilding
        # a repo can NEVER delete a different tenant's identically-named
        # container (container names are ``codegraph-<repo-name>``, shared across
        # tenants that provision the same repo URL). When no tenant is supplied
        # (admin/legacy path) the original unscoped delete is preserved.
        if tenant_id is not None:
            owned = await self.find_container_by_name(name, tenant_id=tenant_id)
            if owned is None:
                raise RuntimeError(f"Container {name} not found")
        headers = optional_sibling_headers(tenant_id) or None
        try:
            await client.delete_workspace(name, headers=headers)
            return True
        except ServiceError as e:
            if e.status_code == 404:
                raise RuntimeError(f"Container {name} not found") from e
            raise

    async def create_container(
        self, request: Any, env_vars: dict[str, str] | None = None, tenant_id: str | None = None
    ) -> ContainerInfo:
        from main import get_workspaces_client

        client = get_workspaces_client()
        # Forward the complete shared request contract. The old hand-built
        # three-field body silently discarded PR provisioning, description,
        # force-rebuild, analysis, user and secret metadata. ``tenant_id`` is
        # intentionally excluded: ownership comes from authenticated headers,
        # never from a caller-controlled JSON field.
        if hasattr(request, "model_dump"):
            payload = request.model_dump(exclude_none=True)
        else:
            payload = {
                "repo_url": str(request.repo_url),
                "branch": request.branch,
            }
        payload.pop("tenant_id", None)
        # ``exact_name`` is a coding-github lookup directive (which EXISTING
        # workspace provision may reuse), not part of the coding-containers
        # create schema, which forbids extra fields: forwarding it 422s EVERY
        # create. Strip it here, where the shared contract is forwarded.
        payload.pop("exact_name", None)

        # ``env_vars`` was a legacy adapter parameter, not part of the
        # coding-containers request schema (extra fields are forbidden). The
        # only clone credential it carried, GIT_TOKEN, has already been copied
        # to request.access_token by ``provision_repository``.
        # Forward the request tenant so the created container is labelled with
        # the caller's tenant (coding-containers reads ``X-Tenant-Id``). Without
        # this the container fell back to the system/default tenant, so later
        # tenant-scoped reads (analyze, proposals, conformance) couldn't see it.
        headers = optional_sibling_headers(tenant_id) or None
        data = self._unwrap_container(await client.create_workspace(headers=headers, **payload))
        return ContainerInfo(**data) if isinstance(data, dict) else data

    async def list_containers(self, tenant_id: str | None = None) -> list[Any]:
        from coding_client.clients import ServiceError

        from main import get_workspaces_client

        client = get_workspaces_client()
        try:
            # coding-containers scopes by the ``X-Tenant-Id`` header, not a query
            # param — passing tenant_id as a param left the list unscoped (and is
            # why coding-analysis' "available containers" lookup reported a wrong
            # set, 404-ing freshly-provisioned containers).
            headers = optional_sibling_headers(tenant_id) or None
            data = await client.list_workspaces(headers=headers)
            items = self._unwrap_containers(data)
            # Re-verify ownership client-side, fail closed. coding-containers'
            # own list filter admits unlabelled containers into EVERY tenant's
            # list, and both routers fall back to a by-name scan over this list
            # when ``find_container_by_name`` returns None — without this filter
            # that fallback re-opens the exact hole the by-name ownership check
            # closes.
            if tenant_id is not None:
                items = [
                    item
                    for item in items
                    if self._container_belongs_to_tenant(item, item, tenant_id)
                ]
            return [ContainerInfo(**item) if isinstance(item, dict) else item for item in items]
        except ServiceError:
            return []


class GitHubProvisioningService:
    """Main orchestration service for GitHub container provisioning."""

    def __init__(
        self,
        docker_service: Any = None,
        github_service: GitHubValidationService | None = None,
    ):
        """Initialize the GitHubProvisioningService service.

        Args:
            docker_service: Container service adapter (auto-created if None)
            github_service: Github validation service
        """
        self.docker_service = docker_service or _ContainersHTTPAdapter()
        self.github_service = github_service
        self.settings = get_settings()

    @staticmethod
    def _require_workspace_repo(existing: Any, requested_repo_url: str) -> None:
        """Raise :class:`WorkspaceRepoMismatchError` unless ``existing`` holds this repo.

        Compared through the adapter's ``_normalize_repo_url`` (trailing ``/``,
        ``.git`` and case ignored), the same rule repo-url reuse matches on. A
        workspace with no ``repo_url`` is refused too: one we cannot attribute is
        not one we hand out.
        """
        name = _ContainersHTTPAdapter._container_field(existing, "container_name")
        have_raw = _ContainersHTTPAdapter._container_field(existing, "repo_url")
        have = _ContainersHTTPAdapter._normalize_repo_url(have_raw)
        want = _ContainersHTTPAdapter._normalize_repo_url(requested_repo_url)
        if have and have == want:
            return
        owner, repo = _owner_repo(requested_repo_url)
        held = f"'{have_raw}'" if have else "no recorded repository"
        logger.warning(
            "Refusing workspace %s (exact-name reuse or force_rebuild removal): "
            "holds %s, request is for %s",
            name,
            held,
            requested_repo_url,
        )
        raise WorkspaceRepoMismatchError(
            f"Workspace '{name}' holds {held}, not the requested '{requested_repo_url}'. "
            "Use a different container_name for this repository.",
            owner=owner,
            repo=repo,
        )

    @staticmethod
    def _require_workspace_toolchain(existing: Any, request: Any) -> None:
        """Raise :class:`WorkspaceToolchainMismatchError` if reuse would change toolchain.

        A no-op when the request does not name a toolchain: an omitted toolchain
        keeps the pre-toolchain behaviour (reuse whatever exists).
        """
        wanted = getattr(request, "toolchain", None)
        if wanted is None:
            return
        have = _ContainersHTTPAdapter._container_field(existing, "toolchain") or _DEFAULT_TOOLCHAIN
        if have == wanted:
            return
        name = _ContainersHTTPAdapter._container_field(existing, "container_name")
        owner, repo = _owner_repo(str(request.repo_url))
        logger.warning(
            "Refusing to reuse workspace %s: toolchain %s, request asks for %s",
            name,
            have,
            wanted,
        )
        raise WorkspaceToolchainMismatchError(
            f"Workspace '{name}' was provisioned with toolchain '{have}', not the requested "
            f"'{wanted}'. Pass force_rebuild=true to rebuild it with '{wanted}' (this "
            "discards the existing workspace), or omit toolchain to reuse it as is.",
            owner=owner,
            repo=repo,
        )

    async def provision_repository(
        self, request: GitHubRepoRequest, tenant_id: str | None = None
    ) -> dict[str, Any]:
        """
        Provision a Docker container for a GitHub repository.

        This method ONLY creates and starts the container. Graph initialization
        is now a separate manual step that must be triggered explicitly via
        the initialization endpoint.

        **Two-Step Process:**
        1. Provision: Creates and starts the container (this method)
        2. Initialize: Builds the code graph (separate manual step)

        This separation allows for:
        - Fast container provisioning
        - Explicit control over when graph analysis runs
        - Better resource management for expensive graph operations

        Args:
            request: GitHub repository provisioning request

        Returns:
            Dict containing container_info and commit_sha
        """
        try:
            logger.info("Starting repository provisioning for %s", request.repo_url)
            if self.github_service is None:
                raise RuntimeError("GitHub validation service is not initialized")

            # If the caller supplied a secret_config_id and no inline token,
            # resolve it via secrets-service *before* validation so the
            # GitHub API probe can use the real PAT. Without this, private
            # repos 404 at validation and never reach the container layer.
            # Same env_vars end up in container env via _maybe_auto_index.
            if request.secret_config_id and not request.access_token:
                resolved = await self._get_environment_variables(request, tenant_id)
                if resolved and "GIT_TOKEN" in resolved:
                    request.access_token = resolved["GIT_TOKEN"]
                    logger.info(
                        "Resolved secret_config_id=%s → GIT_TOKEN, populating access_token",
                        request.secret_config_id,
                    )

            if request.provision_type == "pull_request" and request.pr_number:
                await self.github_service.validate_pull_request_access(request, tenant_id=tenant_id)
                try:
                    commit_sha = await self.github_service.get_pull_request_head_commit(
                        request.repo_url,
                        request.pr_number,
                        request.access_token,
                        tenant_id=tenant_id,
                    )
                except ValueError as pr_err:
                    logger.warning("Could not get PR commit SHA: %s", str(pr_err))
                    commit_sha = None
            else:
                await self.github_service.validate_repository_access(request, tenant_id=tenant_id)
                try:
                    commit_sha = await self.github_service.get_branch_head_commit(
                        request.repo_url,
                        request.branch,
                        request.access_token,
                        tenant_id=tenant_id,
                    )
                except ValueError as branch_err:
                    logger.warning("Could not get commit SHA: %s", str(branch_err))
                    commit_sha = None
            # There is deliberately no KUBERNETES_IN_CLUSTER branch here. One
            # existed and imported ``get_k8s_container_provisioner``, a function
            # defined nowhere, so it could only ever raise ImportError. It was
            # also unreachable: the coding-github pod does not set
            # KUBERNETES_IN_CLUSTER. Container lifecycle in-cluster belongs to
            # coding-containers, reached through ``self.docker_service``.
            repo_name = self._extract_repo_name(str(request.repo_url))
            if request.container_name:
                expected_container_name = request.container_name
            elif request.provision_type == "pull_request" and request.pr_number:
                expected_container_name = f"codegraph-{repo_name}-pr-{request.pr_number}"
            else:
                expected_container_name = f"codegraph-{repo_name}"
            if getattr(request, "exact_name", False) and request.container_name:
                # Exact-name reuse only: never fall back to "the tenant's first
                # workspace for this repo", which may be a different workspace.
                existing = await self.docker_service.find_container_by_name(
                    request.container_name, tenant_id=tenant_id
                )
                if existing:
                    # ⛔ The name is the caller's, the repo is not proven by it. A
                    # tenant-owned workspace of this name may hold a DIFFERENT repo;
                    # returning it would report another checkout as this repo's,
                    # under this repo's commit_sha. Refuse before force_rebuild too,
                    # so a mismatch never deletes the other repo's workspace either.
                    self._require_workspace_repo(existing, str(request.repo_url))
            else:
                existing = await self.docker_service.find_existing_container(
                    str(request.repo_url), tenant_id=tenant_id
                )
            # ⛔ #136: ``existing`` above was matched on the REPO (find_existing_container
            # compares normalised repo_url) or name-and-repo-checked (exact_name).
            # ``expected_container_name`` is matched on NOTHING: it is the caller's
            # container_name or a derived ``codegraph-<repo>[-pr-N]``, and a tenant
            # workspace of that name may hold a DIFFERENT repo (stale or mistyped
            # name, or two owners' repos sharing a repo name). Deleting it by name
            # destroyed another repo's workspace and any card work on it.
            # The name lookup and the repo check run BEFORE ANY removal, including
            # ``existing``'s: a refusal must be atomic, never "this repo's workspace
            # deleted, then 409". An unattributed workspace (no repo_url) is refused
            # too. Nothing found means nothing to remove by name.
            # With exact_name the name WAS the lookup above (``existing``), already
            # repo-checked, so it is not looked up again.
            named = None
            if request.force_rebuild and not (
                getattr(request, "exact_name", False) and request.container_name
            ):
                named = await self.docker_service.find_container_by_name(
                    expected_container_name, tenant_id=tenant_id
                )
                if named:
                    self._require_workspace_repo(named, str(request.repo_url))
                    if existing and _ContainersHTTPAdapter._container_field(
                        named, "container_name"
                    ) == _ContainersHTTPAdapter._container_field(existing, "container_name"):
                        # Same workspace as ``existing``: removed once, below.
                        named = None
            if existing:
                if request.force_rebuild:
                    logger.info("Force rebuild requested for %s", request.repo_url)
                    logger.info(
                        "Removing existing container %s...",
                        existing.container_name,
                    )
                    await self.docker_service.remove_container(existing, tenant_id=tenant_id)
                else:
                    # Before reuse, never after: a mismatch must not hand back the
                    # wrong workspace, and must not rebuild one nobody asked to lose.
                    self._require_workspace_toolchain(existing, request)
                    logger.info(
                        "Repository %s already has container: %s",
                        request.repo_url,
                        existing.container_name,
                    )
                    logger.info(
                        "Container already exists. Use force_rebuild=true to recreate, or use the initialization endpoint to rebuild the graph."
                    )
                    return {"container_info": existing, "commit_sha": commit_sha}
            if named:
                try:
                    await self.docker_service.remove_container_by_name(
                        expected_container_name, tenant_id=tenant_id
                    )
                    logger.info(
                        "Removed existing container by name: %s",
                        expected_container_name,
                    )
                except RuntimeError as ex:
                    logger.debug(
                        "Container %s not found by name: %s",
                        expected_container_name,
                        ex,
                    )

            # Retrieve environment variables from Secrets API if configured.
            # If we already pre-resolved GIT_TOKEN into request.access_token
            # above, skip this call — otherwise env_vars={"GIT_TOKEN": …}
            # would be sent to coding-containers alongside access_token, and
            # coding-containers rejects unknown body fields.
            if request.secret_config_id and request.access_token:
                env_vars = None
            else:
                env_vars = await self._get_environment_variables(request, tenant_id)

            container = await self.docker_service.create_container(
                request, env_vars, tenant_id=tenant_id
            )

            logger.info(
                "Container provisioning completed: %s",
                container.container_name,
            )
            self._maybe_auto_index(container, tenant_id)
            return {"container_info": container, "commit_sha": commit_sha}
        except ValueError as e:
            logger.error("Repository provisioning failed (validation error): %s", str(e))
            return {
                "container_info": None,
                "commit_sha": None,
                "error": str(e),
                "error_type": "validation",
            }
        except NotFoundException:
            logger.info(
                "Repository not found - re-raising NotFoundException to maintain proper HTTP status"
            )
            raise
        except TenantRepoTokenError:
            # A customer repo with no tenant-bound App installation (or fr
            # unreachable): the router maps this to its own status + error_code.
            raise
        except RuntimeError as e:
            logger.error(
                "Repository provisioning failed (runtime error): %s",
                str(e),
                exc_info=True,
            )
            return {
                "container_info": None,
                "commit_sha": None,
                "error": str(e),
                "error_type": "runtime",
            }
        except Exception as e:
            logger.error(
                "Repository provisioning failed (unexpected error): %s",
                str(e),
                exc_info=True,
            )
            return {
                "container_info": None,
                "commit_sha": None,
                "error": str(e),
                "error_type": "system",
            }

    def _maybe_auto_index(self, container: Any, tenant_id: str | None) -> None:
        """Kick off background scans against coding-analysis and coding-security.

        Both run fire-and-forget — the provision response returns as soon as
        the container is up; graphs + dependency reports fill in asynchronously.
        Gated by AUTO_INDEX_ON_PROVISION so CI and unit tests can opt out.
        """
        if os.getenv("AUTO_INDEX_ON_PROVISION", "true").lower() != "true":
            return
        if container is None or not getattr(container, "container_name", None):
            logger.debug("Skipping auto-index: container name unavailable")
            return
        asyncio.create_task(_trigger_full_scan(container.container_name, tenant_id))
        asyncio.create_task(_trigger_dependency_scan(container.container_name, tenant_id))

    def _extract_repo_name(self, repo_url: str) -> str:
        """Extract repository name from URL for consistent naming."""
        try:
            if repo_url.endswith(".git"):
                repo_url = repo_url[:-4]
            parts = repo_url.rstrip("/").split("/")
            if len(parts) >= 2:
                repo_part = parts[-1]
                return repo_part.replace("_", "_").replace("-", "_")
            return "unknown_repo"
        except ValueError as e:
            logger.warning("Failed to extract repo name from %s: %s", repo_url, e)
            return "unknown_repo"

    async def list_containers(self, tenant_id: str | None = None) -> list[ContainerInfo]:
        """List all managed containers, optionally filtered by tenant.

        Args:
            tenant_id: If provided, only return containers belonging to this tenant.
                      If None, returns all containers (for admin use or legacy code).

        Returns:
            List of ContainerInfo objects
        """
        containers = await self.docker_service.list_containers(tenant_id=tenant_id)
        return containers

    async def remove_container(
        self, container_name: str, tenant_id: str | None = None
    ) -> dict[str, Any]:
        """Remove a specific container.

        Args:
            container_name: The container name to remove
            tenant_id: If provided, validates tenant access before removal.
                      Returns error if container belongs to different tenant.

        Returns:
            Dict with status, message, and container_name
        """
        try:
            containers = await self.docker_service.list_containers(tenant_id=tenant_id)
            target_container = None
            for container in containers:
                if container.container_name == container_name:
                    target_container = container
                    break
            if not target_container:
                raise NotFoundException(f"Container {container_name} not found")

            success = await self.docker_service.remove_container(
                target_container, tenant_id=tenant_id
            )
            if success:
                return {
                    "status": "success",
                    "message": f"Container {container_name} removed successfully",
                    "container_name": container_name,
                }
            return {
                "status": "error",
                "message": f"Failed to remove container {container_name}",
                "container_name": container_name,
            }
        except ValueError as e:
            logger.error("Failed to remove container %s: %s", container_name, str(e))
            raise
        except RuntimeError as e:
            logger.error("Failed to remove container %s: %s", container_name, str(e))
            raise

    async def _get_environment_variables(
        self, request: GitHubRepoRequest, tenant_id: str | None = None
    ) -> dict[str, str] | None:
        """Retrieve environment variables from Secrets API.

        Args:
            request: The provisioning request
            tenant_id: Tenant from the authenticated caller (``require_tenant``).
                The ONLY accepted source. ``request.user_id`` is a *user*
                identifier, not a tenant: secrets-service partitions by
                ``X-Tenant-Id``, so passing a user id there addresses a
                partition keyed by a value no registry ever issued as a tenant,
                and the field is typically unset on the provisioning body
                anyway. When there is no tenant the header is omitted and
                secrets-service refuses the fetch — which is the correct answer.

        Returns:
            Dict[str, str]: Environment variables or None if not configured
        """
        # Check if secret_config_id is provided
        if not request.secret_config_id:
            logger.debug("No secret_config_id provided, skipping secret retrieval")
            return None

        try:
            secrets_url = self.settings.secrets_api_url.rstrip("/")
            if not tenant_id:
                # Previously: ``tenant_id or request.user_id or "anonymous"``.
                # That fabricated a tenant twice over. A literal ``"anonymous"``
                # is not a null — it is a perfectly valid partition key, so the
                # fetch SUCCEEDS against a partition nobody owns, and downstream
                # it is indistinguishable from a real tenant that happens to be
                # named that. No audit trail can attribute it. Absent means
                # absent: send no header and let secrets-service refuse.
                logger.warning(
                    "No tenant for secret_config_id=%s; sending no tenant header. "
                    "secrets-service will refuse the fetch and provisioning "
                    "continues without a token (private repos will fail).",
                    request.secret_config_id,
                )
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{secrets_url}/secrets/{request.secret_config_id}",
                    headers=secrets_service_headers(tenant_id),
                )
            if response.status_code == 404:
                logger.warning(
                    "Secret config %s not found in secrets-service; "
                    "provisioning will proceed without token (private repos will fail).",
                    request.secret_config_id,
                )
                return None
            response.raise_for_status()
            payload = response.json()
            secrets_blob = payload.get("secrets") or {}
            env_vars_blob = payload.get("env_vars") or {}
            # Merge: env_vars first, secrets override. Keys used by the
            # init container's git-clone step live under ``secrets`` by
            # convention (GIT_TOKEN); env_vars are for downstream workloads
            # that some day read them inside the pod.
            combined: dict[str, str] = {**env_vars_blob, **secrets_blob}
            if not combined:
                logger.info(
                    "Secret config %s had no env_vars or secrets; nothing to inject.",
                    request.secret_config_id,
                )
                return None
            return combined
        except httpx.HTTPError as exc:
            logger.warning(
                "Secrets-service fetch failed for config_id=%s: %s — "
                "continuing without token (private repos will fail).",
                request.secret_config_id,
                exc,
            )
            return None


def get_github_provisioning_service() -> GitHubProvisioningService:
    """Get GitHub provisioning service instance."""
    from services.github_validation import get_github_validation_service

    return GitHubProvisioningService(
        github_service=get_github_validation_service(),
    )
