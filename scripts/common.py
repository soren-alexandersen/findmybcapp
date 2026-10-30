"""Shared helpers for findmybc.app scripts.

Paths, ID parsing, slugs and stable JSON writing. Standard library only.
"""
import hashlib
import html
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
APPS = DATA / "apps"
NEEDS = DATA / "needs"
REGISTRY = DATA / "registry.json"
PUBLISHER_DATA = ROOT / "publisher-data"
SCHEMA = ROOT / "schema"

ZERO_GUID = "00000000-0000-0000-0000-000000000000"
GUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def parse_app_id(app_id):
    """Split AppSource's compound ID into its parts.

    'PUBID.x|AID.y|PAPPID.z' -> {'PUBID': 'x', 'AID': 'y', 'PAPPID': 'z'}
    Some IDs also carry a leading 'TYPE.connect' part.
    SaaS offers use the short form 'publisher.offer' -> {'PUBID': 'publisher', 'AID': 'offer'}.
    """
    if "|" not in app_id:
        pub, _, offer = app_id.partition(".")
        return {"PUBID": pub, "AID": offer}
    parts = {}
    for piece in app_id.split("|"):
        key, _, value = piece.partition(".")
        parts[key] = value
    return parts


def slugify(text):
    """Readable, path-safe slug: 'Business Express A/S' -> 'business-express-a-s'."""
    text = unicodedata.normalize("NFKD", text or "")
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return re.sub(r"-{2,}", "-", text)[:60].strip("-")


def safe_token(text):
    """Path-safe token for IDs (keeps underscores)."""
    return re.sub(r"[^a-z0-9_-]+", "-", (text or "").lower()).strip("-")


def is_saas_id(app_id):
    """SaaS offers have the short ID form 'publisher.offer' (no PAPPID guid)."""
    return "|" not in app_id


def app_url(app_id, product_type=None):
    """Public Marketplace page for an app. BC apps and SaaS offers live under different paths."""
    if product_type == "SaaS" or is_saas_id(app_id):
        return "https://appsource.microsoft.com/en-us/product/web-apps/" + app_id
    return ("https://appsource.microsoft.com/en-us/product/dynamics-365-business-central/"
            + app_id.replace("|", "%7C"))


def source_hash(raw):
    """Hash of the listing text the mapping is derived from (not the name).

    When this changes, the listing changed and the mapping is stale.
    """
    plans = [(p.get("displayName"), p.get("description")) for p in raw.get("plans") or []]
    payload = json.dumps([
        raw.get("longSummary"), raw.get("description"),
        plans, raw.get("pricingTypes"),
    ], ensure_ascii=False, sort_keys=True)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    """Stable formatting so git diffs stay small and readable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.write_text(text, encoding="utf-8")
    return True


def attr(raw, key):
    """Catalog attributes arrive as JSON encoded in strings; decode when possible."""
    for a in raw.get("attributes") or []:
        if a.get("key") == key:
            value = a.get("value")
            if isinstance(value, str) and value.strip()[:1] in "[{":
                try:
                    return json.loads(value)
                except ValueError:
                    pass
            return value
    return None


def trial_text(value):
    """{'Duration': {'Length': 1, 'Unit': 'Month'}} -> '1 Month'"""
    d = value.get("Duration") if isinstance(value, dict) else None
    return f"{d.get('Length')} {d.get('Unit')}" if d else None


BC_MENTION = re.compile(r"business central|dynamics 365 bc|d365 ?bc|\bbc\b", re.I)


def mentions_bc(raw):
    """True when the title or description names Business Central. NAV never counts.
    A SaaS offer without it is left off the site."""
    text = " ".join(str(raw.get(k) or "") for k in ("displayName", "summary", "longSummary", "description"))
    return bool(BC_MENTION.search(text))


# Listing says the app works wherever Business Central is sold ("Supported countries: All
# countries where Business Central is available"). Stored as a yes/no fact, never as text.
_COUNTRY_HEAD = re.compile(r"supported\s+(?:countr(?:y|ies)|regions|markets)(?:\s*(?:/|and)\s*regions?)?|\b(?:countries|availability)\s*:", re.I)
_ALL_WORD = re.compile(r"\b(?:all|every|any)\b(?!\s+(?:of\s+)?(?:the\s+)?(?:following|listed|below))", re.I)
_ALL_COUNTRIES = re.compile(
    r"\b(?:all|every|any)\s+(?:the\s+)?(?:countries|countries/regions|countries\s+and\s+regions|regions|markets|localizations|localisations)\b"
    r"[^.]{0,80}?(?:business\s+central|\bbc\b|d365|dynamics\s+365)"
    r"|\bavailable\s+(?:in|for|to)\s*:?\s+all\s+(?:countries|regions|markets)\b"
    r"(?![^.]{0,40}\bwhere\b(?![^.]{0,60}(?:business\s+central|\bbc\b|dynamics)))"
    r"|\ball\s+(?:microsoft\s+)?(?:dynamics\s+365\s+)?(?:business\s+central|bc)\s+(?:countries|localizations|localisations|markets)\b", re.I)


def _plain(text):
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def all_countries(raw):
    """True when the listing says it works in every country where Business Central is available."""
    text = _plain(" ".join(str(raw.get(k) or "") for k in ("longSummary", "description")))
    for m in _COUNTRY_HEAD.finditer(text):
        seg = text[m.end():m.end() + 120]
        stop = re.search(r"supported\s+(?:languages?|editions?)", seg, re.I)
        if _ALL_WORD.search(seg[:stop.start()] if stop else seg):
            return True
    return bool(_ALL_COUNTRIES.search(text))


def source_fields(raw):
    """The factual source zone of an app file, from one raw catalog record."""
    return {
        "name": raw.get("displayName") or "",
        "publisher": raw.get("publisherDisplayName") or "",
        "productType": raw.get("productType") or "DynamicsBC",
        "url": app_url(raw["uniqueProductId"], raw.get("productType")),
        "lastModified": raw.get("lastModifiedDateTime") or "",
        "appVersion": attr(raw, "AppVersion"),
        "pricingTypes": sorted(raw.get("pricingTypes") or []),
        "freeTrial": trial_text(attr(raw, "FreeTrialDurationInDays")),
        "helpLink": attr(raw, "HelpLink") or None,
        "sourceHash": source_hash(raw),
        "mentionsBC": mentions_bc(raw),
        "allCountries": all_countries(raw),
    }


def cap(text):
    return text[:1].upper() + text[1:]


def normalize_lang(code):
    base, _, region = code.partition("-")
    return base.lower() + ("-" + region.upper() if region else "")


def mapping_fields(mapped, shash, mapped_on=None):
    """Mapping zone from one mapping record (one mapped app).

    Quotes in the record are dropped on purpose: only our paraphrase is kept.
    'what' is our one-line summary, 'type' the app's shape.
    """
    pricing = mapped.get("pricing") or {}
    return {
        "sourceHash": shash,
        "mappedOn": mapped_on,
        "summary": (mapped.get("what") or "").strip() or None,
        "shape": mapped.get("type") or "other",
        "primary": [{"need": p["group"], "confidence": p.get("confidence", "medium")}
                    for p in mapped.get("primary") or []],
        "secondary": [s["group"] for s in mapped.get("secondary") or []],
        "claims": [c["claim"] for c in mapped.get("capabilities") or [] if c.get("claim")],
        "countries": sorted({c.upper() for c in mapped.get("countries") or []}),
        "languages": sorted({normalize_lang(l) for l in mapped.get("languages") or []}),
        "industries": sorted({cap(i.strip()) for i in mapped.get("industries") or [] if i.strip()}),
        "connectsTo": mapped.get("connects_to") or [],
        "requires": mapped.get("requires") or [],
        "deployment": mapped.get("deployment") or "not_stated",
        "pricing": pricing.get("model", "not_stated") if isinstance(pricing, dict) else pricing,
        "hide": bool(mapped.get("hide")),
        "hideReason": mapped.get("hide_reason") or "",
        "noFit": mapped.get("no_fit") or "",
    }
