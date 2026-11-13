"""
Security regression detector for PR validation.

This module detects potential security regressions and vulnerabilities
introduced by code changes, providing context-aware security analysis.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from coding_client.models.pr_validation import (
    CodebaseContext,
    FileChange,
    FileType,
    ValidationIssue,
)

logger = logging.getLogger(__name__)


@dataclass
class PRSecurityPattern:
    """Represents a security pattern to detect in PR validation context."""

    pattern_id: str
    name: str
    pattern: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    description: str
    recommendation: str
    file_types: list[str] = field(default_factory=list)
    context_required: bool = False
    cwe_id: str | None = None


@dataclass
class SecurityVulnerability:
    """Represents a detected security vulnerability."""

    vuln_type: str
    severity: str
    file_path: str
    line_number: int
    code_snippet: str
    description: str
    recommendation: str
    confidence: float
    cwe_id: str | None = None


@dataclass
class SecurityContext:
    """Represents security context for analysis."""

    authentication_methods: list[str] = field(default_factory=list)
    authorization_checks: list[str] = field(default_factory=list)
    input_validation: list[str] = field(default_factory=list)
    output_encoding: list[str] = field(default_factory=list)
    crypto_usage: list[str] = field(default_factory=list)
    sensitive_data_handling: list[str] = field(default_factory=list)


class SecurityRegressionDetector:
    """
    Detects security regressions and vulnerabilities in code changes.

    This detector focuses on identifying:
    - Injection vulnerabilities (SQL, Command, XSS)
    - Authentication and authorization bypasses
    - Cryptographic weaknesses
    - Input validation issues
    - Insecure data handling
    - Security control removals
    """

    def __init__(self):
        self.security_patterns = self._initialize_security_patterns()
        self.sensitive_functions = self._initialize_sensitive_functions()
        self.crypto_patterns = self._initialize_crypto_patterns()
        self.input_patterns = self._initialize_input_patterns()

    def _initialize_security_patterns(self) -> list[PRSecurityPattern]:
        """Initialize security pattern definitions."""
        return [
            # SQL Injection patterns
            PRSecurityPattern(
                pattern_id="sql_injection_1",
                name="SQL Injection - String Concatenation",
                pattern=r'(SELECT|INSERT|UPDATE|DELETE).*\+.*[\'"]',
                severity="high",
                description="Potential SQL injection via string concatenation",
                recommendation="Use parameterized queries or prepared statements",
                file_types=["python", "javascript", "typescript"],
                cwe_id="CWE-89",
            ),
            PRSecurityPattern(
                pattern_id="sql_injection_2",
                name="SQL Injection - Format String",
                pattern=r"(SELECT|INSERT|UPDATE|DELETE).*\.format\s*\(",
                severity="high",
                description="Potential SQL injection via string formatting",
                recommendation="Use parameterized queries instead of string formatting",
                file_types=["python"],
                cwe_id="CWE-89",
            ),
            # Command Injection patterns
            PRSecurityPattern(
                pattern_id="command_injection_1",
                name="Command Injection - os.system",
                pattern=r'os\.system\s*\([^)]*[\'"].*\+',
                severity="critical",
                description="Command injection via os.system with concatenated input",
                recommendation="Use subprocess with shell=False and validate inputs",
                file_types=["python"],
                cwe_id="CWE-78",
            ),
            PRSecurityPattern(
                pattern_id="command_injection_2",
                name="Command Injection - subprocess with shell",
                pattern=r"subprocess\.(call|run|Popen).*shell\s*=\s*True",
                severity="high",
                description="Command execution with shell=True can be dangerous",
                recommendation="Use shell=False and validate all inputs",
                file_types=["python"],
                cwe_id="CWE-78",
            ),
            # XSS patterns
            PRSecurityPattern(
                pattern_id="xss_1",
                name="XSS - innerHTML with user input",
                pattern=r'\.innerHTML\s*=.*[^\'"]',
                severity="high",
                description="Potential XSS via innerHTML assignment",
                recommendation="Use textContent or properly escape HTML",
                file_types=["javascript", "typescript"],
                cwe_id="CWE-79",
            ),
            # Authentication bypass patterns
            PRSecurityPattern(
                pattern_id="auth_bypass_1",
                name="Authentication Bypass - Commented Auth",
                pattern=r"#.*auth|//.*auth|/\*.*auth.*\*/",
                severity="high",
                description="Authentication code appears to be commented out",
                recommendation="Ensure authentication checks are not disabled",
                file_types=["python", "javascript", "typescript"],
            ),
            # Weak cryptography patterns
            PRSecurityPattern(
                pattern_id="weak_crypto_1",
                name="Weak Cryptography - MD5",
                pattern=r"hashlib\.md5|crypto\.md5|MD5",
                severity="medium",
                description="MD5 is cryptographically broken",
                recommendation="Use SHA-256 or stronger hashing algorithms",
                file_types=["python", "javascript", "typescript"],
                cwe_id="CWE-327",
            ),
            PRSecurityPattern(
                pattern_id="weak_crypto_2",
                name="Weak Cryptography - SHA1",
                pattern=r"hashlib\.sha1|crypto\.sha1|SHA1",
                severity="medium",
                description="SHA1 is deprecated for security purposes",
                recommendation="Use SHA-256 or stronger hashing algorithms",
                file_types=["python", "javascript", "typescript"],
                cwe_id="CWE-327",
            ),
            # Input validation patterns
            PRSecurityPattern(
                pattern_id="input_validation_1",
                name="Missing Input Validation - eval",
                pattern=r"\beval\s*\(",
                severity="critical",
                description="Use of eval() can execute arbitrary code",
                recommendation="Avoid eval() or strictly validate input",
                file_types=["python", "javascript", "typescript"],
                cwe_id="CWE-95",
            ),
            PRSecurityPattern(
                pattern_id="input_validation_2",
                name="Missing Input Validation - exec",
                pattern=r"\bexec\s*\(",
                severity="critical",
                description="Use of exec() can execute arbitrary code",
                recommendation="Avoid exec() or strictly validate input",
                file_types=["python"],
                cwe_id="CWE-95",
            ),
            # Hardcoded secrets patterns
            PRSecurityPattern(
                pattern_id="hardcoded_secret_1",
                name="Hardcoded Secret - Password",
                pattern=r'password\s*=\s*[\'"][^\'"\s]{8,}[\'"]',
                severity="high",
                description="Hardcoded password detected",
                recommendation="Use environment variables or secure configuration",
                file_types=["python", "javascript", "typescript"],
                cwe_id="CWE-798",
            ),
            PRSecurityPattern(
                pattern_id="hardcoded_secret_2",
                name="Hardcoded Secret - API Key",
                pattern=r'(api_key|apikey|secret_key)\s*=\s*[\'"][a-zA-Z0-9]{20,}[\'"]',
                severity="high",
                description="Hardcoded API key or secret detected",
                recommendation="Use environment variables or secure configuration",
                file_types=["python", "javascript", "typescript"],
                cwe_id="CWE-798",
            ),
            # CSRF patterns
            PRSecurityPattern(
                pattern_id="csrf_1",
                name="CSRF - Missing CSRF Protection",
                pattern=r"@app\.route.*methods.*POST.*(?!.*csrf)",
                severity="medium",
                description="POST endpoint without apparent CSRF protection",
                recommendation="Implement CSRF protection for state-changing operations",
                file_types=["python"],
            ),
            # Insecure random patterns
            PRSecurityPattern(
                pattern_id="insecure_random_1",
                name="Insecure Random - random module",
                pattern=r"import\s+random|from\s+random\s+import",
                severity="low",
                description="Python random module is not cryptographically secure",
                recommendation="Use secrets module for security-sensitive randomness",
                file_types=["python"],
                cwe_id="CWE-338",
            ),
        ]

    def _initialize_sensitive_functions(self) -> dict[str, list[str]]:
        """Initialize lists of security-sensitive functions."""
        return {
            "python": [
                "eval",
                "exec",
                "compile",
                "__import__",
                "os.system",
                "os.popen",
                "os.execv",
                "subprocess.call",
                "subprocess.run",
                "subprocess.Popen",
                "pickle.loads",
                "pickle.load",
                "cPickle.loads",
                "input",
                "raw_input",
            ],
            "javascript": [
                "eval",
                "setTimeout",
                "setInterval",
                "Function",
                "document.write",
                "document.writeln",
                "innerHTML",
                "outerHTML",
            ],
        }

    def _initialize_crypto_patterns(self) -> dict[str, list[str]]:
        """Initialize cryptographic function patterns."""
        return {
            "weak_hashing": [r"md5", r"sha1", r"crc32"],
            "weak_encryption": [r"des", r"3des", r"rc4"],
            "good_hashing": [
                r"sha256",
                r"sha512",
                r"bcrypt",
                r"scrypt",
                r"argon2",
            ],
            "good_encryption": [r"aes", r"chacha20", r"rsa"],
        }

    def _initialize_input_patterns(self) -> dict[str, list[str]]:
        """Initialize input validation patterns."""
        return {
            "user_input_sources": [
                r"request\.args",
                r"request\.form",
                r"request\.json",
                r"input\s*\(",
                r"raw_input\s*\(",
                r"req\.body",
                r"req\.query",
                r"req\.params",
            ],
            "dangerous_sinks": [
                r"eval\s*\(",
                r"exec\s*\(",
                r"os\.system\s*\(",
                r"subprocess\.",
                r"document\.write\s*\(",
                r"innerHTML\s*=",
            ],
        }

    async def detect_security_regressions(
        self, changes: list[FileChange], codebase_context: CodebaseContext
    ) -> list[ValidationIssue]:
        """
        Detect security regressions in code changes.

        Args:
            changes: List of file changes to analyze
            codebase_context: Context about the codebase

        Returns:
            List of ValidationIssue objects for security concerns
        """
        issues: list[ValidationIssue] = []

        try:
            for change in changes:
                if self._should_analyze_file(change):
                    # Analyze for security patterns
                    pattern_issues = await self._analyze_security_patterns(change)
                    issues.extend(pattern_issues)

                    # Analyze for removed security controls
                    removal_issues = await self._analyze_security_removals(change)
                    issues.extend(removal_issues)

                    # Analyze for data flow issues
                    dataflow_issues = await self._analyze_data_flow_security(change)
                    issues.extend(dataflow_issues)

                    # Analyze for authentication/authorization issues
                    auth_issues = await self._analyze_auth_security(change)
                    issues.extend(auth_issues)

        except Exception as e:
            logger.error(f"Error in security regression detection: {e}")
            issues.append(
                ValidationIssue(
                    type="security_analysis_error",
                    severity="medium",
                    message=f"Security analysis failed: {str(e)[:100]}",
                    file="system",
                    rule="security_detector_error",
                    suggestion="Manual security review recommended",
                )
            )

        return issues

    def _should_analyze_file(self, change: FileChange) -> bool:
        """Determine if a file should be analyzed for security issues."""
        return (
            change.file_type in [FileType.PYTHON, FileType.JAVASCRIPT, FileType.TYPESCRIPT]
            and not change.binary
            and bool(change.before_content or change.after_content)
        )

    async def _analyze_security_patterns(self, change: FileChange) -> list[ValidationIssue]:
        """Analyze file for known security vulnerability patterns."""
        issues: list[ValidationIssue] = []
        content = change.after_content or change.before_content or ""

        if not content:
            return issues

        file_type = (
            change.file_type.value if hasattr(change.file_type, "value") else str(change.file_type)
        )

        # Check each security pattern
        for pattern in self.security_patterns:
            if file_type in pattern.file_types or not pattern.file_types:
                matches = self._find_pattern_matches(content, pattern, change.filename)
                for match in matches:
                    issues.append(
                        ValidationIssue(
                            type=f"security_pattern_{pattern.pattern_id}",
                            severity=pattern.severity,
                            message=f"{pattern.name}: {pattern.description}",
                            file=change.filename,
                            line=match["line_number"],
                            rule=pattern.pattern_id,
                            suggestion=pattern.recommendation,
                        )
                    )

        return issues

    def _find_pattern_matches(
        self, content: str, pattern: PRSecurityPattern, filename: str
    ) -> list[dict[str, Any]]:
        """Find matches for a security pattern in content."""
        matches = []
        lines = content.split("\n")

        for line_num, line in enumerate(lines, 1):
            if re.search(pattern.pattern, line, re.IGNORECASE):
                # Additional context checks if required
                if pattern.context_required:
                    if not self._validate_pattern_context(line, lines, line_num, pattern):
                        continue

                matches.append(
                    {
                        "line_number": line_num,
                        "line_content": line.strip(),
                        "pattern_id": pattern.pattern_id,
                    }
                )

        return matches

    def _validate_pattern_context(
        self,
        line: str,
        all_lines: list[str],
        line_num: int,
        pattern: PRSecurityPattern,
    ) -> bool:
        """Validate additional context for pattern matches with sophisticated analysis."""

        # Augmented context validation with multiple layers of analysis

        # 1. Basic filtering for comments and strings
        if self._is_in_comment_or_string(line, all_lines, line_num):
            return False

        # 2. Pattern-specific context validation
        if pattern.pattern_id == "sql_injection_1":
            return self._validate_sql_injection_context(line, all_lines, line_num)
        elif pattern.pattern_id.startswith("xss_"):
            return self._validate_xss_context(line, all_lines, line_num)
        elif pattern.pattern_id.startswith("auth_"):
            return self._validate_auth_context(line, all_lines, line_num)
        elif pattern.pattern_id.startswith("crypto_"):
            return self._validate_crypto_context(line, all_lines, line_num)

        # 3. General vulnerability context validation
        return self._validate_general_vulnerability_context(line, all_lines, line_num, pattern)

    def _is_in_comment_or_string(self, line: str, all_lines: list[str], line_num: int) -> bool:
        """Check if the pattern match is within a comment or string literal."""
        stripped = line.strip()

        # Single-line comments
        if stripped.startswith("#") or stripped.startswith("//") or stripped.startswith("*"):
            return True

        # Check for inline comments
        if " # " in line or " // " in line:
            # Find position of pattern match and comment
            line.find(" # ") if " # " in line else line.find(" // ")
            # If pattern is after comment, it's probably in a comment
            return True  # Conservative approach

        # Multi-line comment blocks (/* */, """ """, etc.)
        if self._is_in_multiline_comment(all_lines, line_num):
            return True

        # String literals - streamlined check
        quote_count_single = line.count("'")
        quote_count_double = line.count('"')

        # If odd number of quotes, likely inside a string
        if (quote_count_single % 2 == 1) or (quote_count_double % 2 == 1):
            return True

        return False

    def _is_in_multiline_comment(self, all_lines: list[str], line_num: int) -> bool:
        """Check if current line is within a multi-line comment block."""
        # Look backwards for comment start
        comment_start_patterns = ["/*", '"""', "'''"]
        comment_end_patterns = ["*/", '"""', "'''"]

        in_comment = False
        for i in range(max(0, line_num - 20), line_num):  # Check last 20 lines
            if i < len(all_lines):
                line = all_lines[i]
                for start_pattern in comment_start_patterns:
                    if start_pattern in line:
                        in_comment = True
                        break
                for end_pattern in comment_end_patterns:
                    if end_pattern in line and in_comment:
                        in_comment = False
                        break

        return in_comment

    def _validate_sql_injection_context(
        self, line: str, all_lines: list[str], line_num: int
    ) -> bool:
        """Validate SQL injection pattern context."""
        line_lower = line.lower()

        # Check for legitimate SQL usage patterns
        safe_patterns = [
            "select * from test",  # Test data
            "create table",  # Schema definition
            "insert into test",  # Test data
            "pragma table_info",  # SQLite introspection
            "show tables",  # MySQL introspection
        ]

        for safe_pattern in safe_patterns:
            if safe_pattern in line_lower:
                return False

        # Check surrounding context for SQL injection indicators
        context_window = 3
        context_lines = all_lines[max(0, line_num - context_window) : line_num + context_window + 1]
        context = "\n".join(context_lines).lower()

        # Look for dynamic query construction
        dangerous_indicators = [
            "user_input",
            "request.form",
            "request.args",
            "params",
            "format(",
            " + ",
            ".format(",
            '"',
        ]

        for indicator in dangerous_indicators:
            if indicator in context:
                return True  # Likely vulnerable

        # Check for prepared statement usage (safe)
        safe_indicators = [
            "preparedstatement",
            "parameterized",
            "bind_param",
            "execute(",
            "?",
        ]

        for indicator in safe_indicators:
            if indicator in context:
                return False  # Likely safe

        return True  # Default to flagging as potential issue

    def _validate_xss_context(self, line: str, all_lines: list[str], line_num: int) -> bool:
        """Validate XSS pattern context."""
        line_lower = line.lower()

        # Check for safe output encoding
        safe_patterns = [
            "escape(",
            "html.escape",
            "cgi.escape",
            "bleach.clean",
            "sanitize",
            "encode(",
        ]

        for safe_pattern in safe_patterns:
            if safe_pattern in line_lower:
                return False

        # Check for dangerous output contexts
        dangerous_patterns = [
            "innerhtml",
            "document.write",
            "eval(",
            "onclick=",
            "onerror=",
            "javascript:",
        ]

        for dangerous_pattern in dangerous_patterns:
            if dangerous_pattern in line_lower:
                return True

        return True  # Conservative approach for XSS

    def _validate_auth_context(self, line: str, all_lines: list[str], line_num: int) -> bool:
        """Validate authentication pattern context."""
        line_lower = line.lower()

        # Check for test or example code
        test_indicators = ["test", "example", "demo", "mock", "fake", "dummy"]

        for indicator in test_indicators:
            if indicator in line_lower:
                return False

        # Check for proper authentication patterns
        proper_auth_patterns = [
            "bcrypt",
            "scrypt",
            "pbkdf2",
            "argon2",
            "hash_password",
            "verify_password",
        ]

        context_window = 5
        context_lines = all_lines[max(0, line_num - context_window) : line_num + context_window + 1]
        context = "\n".join(context_lines).lower()

        for pattern in proper_auth_patterns:
            if pattern in context:
                return False  # Proper authentication detected

        return True  # Potential authentication issue

    def _validate_crypto_context(self, line: str, all_lines: list[str], line_num: int) -> bool:
        """Validate cryptography pattern context."""
        line_lower = line.lower()

        # Check for weak crypto usage
        weak_crypto = ["md5", "sha1", "des", "rc4", "base64"]  # Not encryption

        for weak in weak_crypto:
            if weak in line_lower:
                return True  # Weak crypto usage

        # Check for proper crypto libraries
        strong_crypto = [
            "cryptography",
            "pycrypto",
            "nacl",
            "aes",
            "rsa",
            "ecdsa",
        ]

        for strong in strong_crypto:
            if strong in line_lower:
                return False  # Strong crypto detected

        return True  # Default to flagging crypto issues

    def _validate_general_vulnerability_context(
        self,
        line: str,
        all_lines: list[str],
        line_num: int,
        pattern: PRSecurityPattern,
    ) -> bool:
        """General vulnerability context validation."""
        line.lower()

        # Check for test files or example code
        test_file_indicators = [
            "test_",
            "_test",
            "example",
            "demo",
            "tutorial",
            "sample",
        ]

        # Get context including file information if available
        context_window = 2
        context_lines = all_lines[max(0, line_num - context_window) : line_num + context_window + 1]
        context = "\n".join(context_lines).lower()

        for indicator in test_file_indicators:
            if indicator in context:
                return False  # Likely test/example code

        # Check for mitigation patterns around the vulnerability
        mitigation_patterns = [
            "validate",
            "sanitize",
            "escape",
            "filter",
            "whitelist",
            "blacklist",
            "check",
            "verify",
        ]

        for mitigation in mitigation_patterns:
            if mitigation in context:
                return False  # Mitigation detected

        # Check pattern severity for validation threshold
        if pattern.severity in ["low", "info"]:
            # Be more lenient with low severity patterns
            return len(context.split()) > 10  # Only flag if substantial code

        return True  # Default to flagging the issue

    async def _analyze_security_removals(self, change: FileChange) -> list[ValidationIssue]:
        """Analyze removed lines for security control removals."""
        issues: list[ValidationIssue] = []

        if not change.patch:
            return issues

        patch_lines = change.patch.split("\n")
        removed_lines = [
            line[1:] for line in patch_lines if line.startswith("-") and not line.startswith("---")
        ]

        security_keywords = [
            "authenticate",
            "authorize",
            "permission",
            "access_control",
            "validate",
            "sanitize",
            "escape",
            "csrf_token",
            "rate_limit",
            "throttle",
            "security_check",
        ]

        for line in removed_lines:
            line_lower = line.lower()
            for keyword in security_keywords:
                if keyword in line_lower:
                    issues.append(
                        ValidationIssue(
                            type="security_control_removal",
                            severity="high",
                            message=f"Potential security control removal: {keyword}",
                            file=change.filename,
                            rule="security_removal_check",
                            suggestion=f"Verify that removing '{keyword}' doesn't weaken security: {line.strip()}",
                        )
                    )
                    break

        return issues

    async def _analyze_data_flow_security(self, change: FileChange) -> list[ValidationIssue]:
        """Analyze data flow for security issues."""
        issues: list[ValidationIssue] = []
        content = change.after_content or change.before_content or ""

        if not content:
            return issues

        # Find user input sources and dangerous sinks
        input_sources = self._find_input_sources(content)
        dangerous_sinks = self._find_dangerous_sinks(content)

        # Look for potential taint flow (streamlined analysis)
        if input_sources and dangerous_sinks:
            # Check if input flows to dangerous sinks without validation
            validation_checks = self._find_validation_checks(content)

            if not validation_checks:
                issues.append(
                    ValidationIssue(
                        type="unsafe_data_flow",
                        severity="high",
                        message="User input may flow to dangerous sink without validation",
                        file=change.filename,
                        rule="data_flow_analysis",
                        suggestion="Add input validation between user input and dangerous operations",
                    )
                )

        return issues

    def _find_input_sources(self, content: str) -> list[dict[str, Any]]:
        """Find user input sources in code."""
        sources = []
        lines = content.split("\n")

        for line_num, line in enumerate(lines, 1):
            for pattern in self.input_patterns["user_input_sources"]:
                if re.search(pattern, line):
                    sources.append(
                        {
                            "line_number": line_num,
                            "line_content": line.strip(),
                            "pattern": pattern,
                        }
                    )

        return sources

    def _find_dangerous_sinks(self, content: str) -> list[dict[str, Any]]:
        """Find dangerous sinks in code."""
        sinks = []
        lines = content.split("\n")

        for line_num, line in enumerate(lines, 1):
            for pattern in self.input_patterns["dangerous_sinks"]:
                if re.search(pattern, line):
                    sinks.append(
                        {
                            "line_number": line_num,
                            "line_content": line.strip(),
                            "pattern": pattern,
                        }
                    )

        return sinks

    def _find_validation_checks(self, content: str) -> list[str]:
        """Find input validation checks in code."""
        validation_patterns = [
            r"validate",
            r"sanitize",
            r"escape",
            r"clean",
            r"isinstance\s*\(",
            r"len\s*\(.*\)\s*[<>]",
            r"\.strip\s*\(\)",
            r"\.replace\s*\(",
            r"if\s+.*\s+in\s+",
            r"if\s+.*\.isdigit\s*\(\)",
        ]

        validations = []
        for pattern in validation_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                validations.append(pattern)

        return validations

    async def _analyze_auth_security(self, change: FileChange) -> list[ValidationIssue]:
        """Analyze authentication and authorization security."""
        issues: list[ValidationIssue] = []

        # Check if this is an authentication/authorization related file
        filename_lower = change.filename.lower()
        auth_indicators = ["auth", "login", "permission", "access", "security"]

        if not any(indicator in filename_lower for indicator in auth_indicators):
            return issues

        content = change.after_content or change.before_content or ""
        if not content:
            return issues

        # Check for common auth issues
        auth_issues = [
            (r'if\s+.*==\s*[\'"]admin[\'"]', "Hardcoded admin check", "high"),
            (
                r'password\s*==\s*[\'"]',
                "Hardcoded password comparison",
                "high",
            ),
            (r"session\[.*\]\s*=\s*True", "Basic session flag", "medium"),
            (
                r"authenticated\s*=\s*True",
                "Basic authentication flag",
                "medium",
            ),
        ]

        lines = content.split("\n")
        for line_num, line in enumerate(lines, 1):
            for pattern, description, severity in auth_issues:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append(
                        ValidationIssue(
                            type="auth_security_issue",
                            severity=severity,
                            message=f"Authentication security issue: {description}",
                            file=change.filename,
                            line=line_num,
                            rule="auth_security_check",
                            suggestion=f"Review authentication logic: {line.strip()}",
                        )
                    )

        return issues

    def _calculate_security_risk_score(self, issues: list[ValidationIssue]) -> float:
        """Calculate overall security risk score."""
        if not issues:
            return 0.0

        severity_weights = {
            "critical": 10.0,
            "high": 5.0,
            "medium": 2.0,
            "low": 1.0,
        }

        total_score = 0.0
        for issue in issues:
            weight = severity_weights.get(issue.severity, 1.0)
            total_score += weight

        # Normalize to 0-10 scale
        return min(total_score / len(issues), 10.0)

    def generate_security_recommendations(self, issues: list[ValidationIssue]) -> list[str]:
        """Generate security recommendations based on found issues."""
        recommendations = []

        # Group issues by type
        issue_types: dict[str, list[ValidationIssue]] = {}
        for issue in issues:
            issue_type = issue.type
            if issue_type not in issue_types:
                issue_types[issue_type] = []
            issue_types[issue_type].append(issue)

        # Generate recommendations
        if any("sql_injection" in issue_type for issue_type in issue_types):
            recommendations.append("Implement parameterized queries to prevent SQL injection")

        if any("command_injection" in issue_type for issue_type in issue_types):
            recommendations.append("Use subprocess with shell=False and validate all inputs")

        if any("xss" in issue_type for issue_type in issue_types):
            recommendations.append("Implement proper output encoding and CSP headers")

        if any("hardcoded_secret" in issue_type for issue_type in issue_types):
            recommendations.append("Move secrets to environment variables or secure configuration")

        if any("auth" in issue_type for issue_type in issue_types):
            recommendations.append("Review authentication and authorization logic")

        # General recommendations
        if issues:
            recommendations.append("Consider using automated security scanning tools")
            recommendations.append("Implement security code review practices")
            recommendations.append("Add security-focused unit tests")

        return recommendations
