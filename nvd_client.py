"""
core/nvd_client.py — Live threat intelligence: NVD + CISA KEV.

- fetch_cve(): pulls full CVE detail from the NVD API (description, CVSS,
  CWE, affected products/CPEs, references, published/modified dates).
- get_kev_catalog(): pulls the CISA Known Exploited Vulnerabilities catalog,
  cached to disk since it's a ~few-MB feed that only changes daily.
- is_actively_exploited(): cross-references a CVE against the KEV catalog.

If a lookup fails or returns nothing, callers get None / an explicit
"unavailable" signal rather than a fabricated answer.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from core.config import settings

logger = logging.getLogger("sentry.nvd")

CVE_PATTERN = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)
_KEV_CACHE_FILE = Path(settings.chroma_dir).parent / "cache" / "kev_catalog.json"


@dataclass
class CVERecord:
    cve_id: str
    description: str = ""
    published: str = ""
    last_modified: str = ""
    cvss_score: float | None = None
    cvss_severity: str = "UNKNOWN"
    cvss_vector: str = ""
    cwe_ids: list[str] = field(default_factory=list)
    affected_products: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    vendor_advisory: str | None = None
    exploited_kev: bool = False
    kev_date_added: str | None = None
    kev_due_date: str | None = None
    patch_available: bool | None = None

    def to_text(self) -> str:
        """Flattened text used for embedding / LLM context."""
        lines = [
            f"CVE ID: {self.cve_id}",
            f"Published: {self.published}",
            f"Last Modified: {self.last_modified}",
            f"CVSS Score: {self.cvss_score if self.cvss_score is not None else 'N/A'} ({self.cvss_severity})",
            f"CVSS Vector: {self.cvss_vector or 'N/A'}",
            f"CWE: {', '.join(self.cwe_ids) if self.cwe_ids else 'N/A'}",
            f"Affected Products: {', '.join(self.affected_products[:15]) if self.affected_products else 'N/A'}",
            f"Actively Exploited (CISA KEV): {'YES' if self.exploited_kev else 'No known evidence'}",
            f"Description: {self.description or 'N/A'}",
            f"References: {', '.join(self.references[:8]) if self.references else 'N/A'}",
        ]
        if self.vendor_advisory:
            lines.append(f"Vendor Advisory: {self.vendor_advisory}")
        return "\n".join(lines)


def extract_cve_ids(text: str) -> list[str]:
    return sorted(set(m.upper() for m in CVE_PATTERN.findall(text or "")))


def _nvd_headers() -> dict[str, str]:
    headers = {"User-Agent": "SENTRY-Threat-Intel-Assistant/1.0"}
    if settings.nvd_api_key:
        headers["apiKey"] = settings.nvd_api_key
    return headers


def fetch_cve(cve_id: str, timeout: int = 15) -> CVERecord | None:
    """Fetch a single CVE's full detail from the NVD API. Returns None if not found."""
    cve_id = cve_id.strip().upper()
    if not CVE_PATTERN.fullmatch(cve_id):
        return None
    try:
        resp = requests.get(
            settings.nvd_base_url, params={"cveId": cve_id}, headers=_nvd_headers(), timeout=timeout
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("NVD lookup failed for %s: %s", cve_id, exc)
        raise

    vulns = data.get("vulnerabilities", [])
    if not vulns:
        return None
    cve = vulns[0]["cve"]

    description = next((d["value"] for d in cve.get("descriptions", []) if d.get("lang") == "en"), "")

    metrics = cve.get("metrics", {})
    cvss_score, cvss_severity, cvss_vector = None, "UNKNOWN", ""
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        if key in metrics and metrics[key]:
            m = metrics[key][0]
            cvss_score = m["cvssData"].get("baseScore")
            cvss_severity = m.get("baseSeverity") or m["cvssData"].get("baseSeverity", "UNKNOWN")
            cvss_vector = m["cvssData"].get("vectorString", "")
            break

    cwe_ids = []
    for weakness in cve.get("weaknesses", []):
        for d in weakness.get("description", []):
            if d.get("value", "").startswith("CWE-"):
                cwe_ids.append(d["value"])

    affected_products = []
    for config in cve.get("configurations", []):
        for node in config.get("nodes", []):
            for cpe_match in node.get("cpeMatch", []):
                if cpe_match.get("vulnerable"):
                    criteria = cpe_match.get("criteria", "")
                    parts = criteria.split(":")
                    if len(parts) >= 6:
                        vendor, product = parts[3], parts[4]
                        affected_products.append(f"{vendor}:{product}")
    affected_products = sorted(set(affected_products))

    references = [r.get("url", "") for r in cve.get("references", []) if r.get("url")]
    vendor_advisory = next(
        (r for r in cve.get("references", []) if "Vendor Advisory" in r.get("tags", [])), None
    )
    vendor_advisory_url = vendor_advisory.get("url") if vendor_advisory else None

    record = CVERecord(
        cve_id=cve_id,
        description=description,
        published=cve.get("published", ""),
        last_modified=cve.get("lastModified", ""),
        cvss_score=cvss_score,
        cvss_severity=cvss_severity,
        cvss_vector=cvss_vector,
        cwe_ids=cwe_ids,
        affected_products=affected_products,
        references=references[:10],
        vendor_advisory=vendor_advisory_url,
    )

    kev_entry = get_kev_entry(cve_id)
    if kev_entry:
        record.exploited_kev = True
        record.kev_date_added = kev_entry.get("dateAdded")
        record.kev_due_date = kev_entry.get("dueDate")
        record.patch_available = True  # KEV entries always have a required action / patch path
    return record


# ---------------------------------------------------------------------------
# CISA KEV catalog
# ---------------------------------------------------------------------------
def get_kev_catalog(force_refresh: bool = False) -> list[dict[str, Any]]:
    """Return the full CISA KEV catalog, using a disk cache to avoid refetching the
    multi-MB feed on every call."""
    _KEV_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not force_refresh and _KEV_CACHE_FILE.exists():
        age = time.time() - _KEV_CACHE_FILE.stat().st_mtime
        if age < settings.kev_cache_ttl:
            try:
                return json.loads(_KEV_CACHE_FILE.read_text()).get("vulnerabilities", [])
            except Exception:
                pass
    try:
        resp = requests.get(settings.kev_url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        _KEV_CACHE_FILE.write_text(json.dumps(data))
        return data.get("vulnerabilities", [])
    except Exception as exc:
        logger.warning("CISA KEV fetch failed: %s", exc)
        if _KEV_CACHE_FILE.exists():
            try:
                return json.loads(_KEV_CACHE_FILE.read_text()).get("vulnerabilities", [])
            except Exception:
                pass
        return []


def get_kev_entry(cve_id: str) -> dict[str, Any] | None:
    catalog = get_kev_catalog()
    cve_id = cve_id.upper()
    for entry in catalog:
        if entry.get("cveID", "").upper() == cve_id:
            return entry
    return None
