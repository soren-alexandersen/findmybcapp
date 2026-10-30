"""Build the site's data from the repo.

Reads data/apps, data/needs and publisher-data. Writes site/data/:
  apps.json        visible apps in the compact shape app.js reads
  groups.json      needs with their questions
  statements.json  routing statements as [need index, text]
  vectors.bin      statement embeddings (bge-small), rebuilt only when statements change

Effective app values = mapping, then curated.overrides on top, then publisher
supplements added (countries, languages, extends, worksWellWith, consented logo).

Run:  python3 scripts/build.py            (--no-vectors to skip embeddings)
"""
import argparse
import hashlib
import json
import sys
from collections import Counter

from common import APPS, NEEDS, PUBLISHER_DATA, REGISTRY, ROOT, read_json, write_json

SITE_DATA = ROOT / "site" / "data"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def works_with(app, labels):
    """A SaaS offer that names BC in its listing works with Business Central.
    BC apps don't need saying."""
    src = app["source"]
    if src.get("productType") == "SaaS" and src.get("mentionsBC") \
            and "Business Central" not in labels:
        return ["Business Central"] + labels
    return labels


def effective_mapping(app):
    m = dict(app["mapping"])
    m.update(app["curated"].get("overrides") or {})
    return m


def load_publisher_files(registry):
    out = {}
    for aid, entry in registry["apps"].items():
        f = PUBLISHER_DATA / entry["publisherFolder"] / f"{entry['stem']}.json"
        if f.exists():
            out[aid] = read_json(f)
    return out


def merge(base, extra):
    """Base list plus publisher additions, order kept, no duplicates."""
    seen, out = set(), []
    for x in list(base) + list(extra):
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def label_key(text):
    """Comparison key for free-text labels: case and spacing never make a new value."""
    return " ".join(text.split()).casefold()


def canonical_labels(values):
    """One spelling per label key across the catalog: the most used spelling wins,
    ties go to the alphabetically first, so the choice is stable between builds."""
    seen = {}
    for v in values:
        v = " ".join(v.split())
        if v:
            seen.setdefault(label_key(v), Counter())[v] += 1
    return {k: min(c, key=lambda s: (-c[s], s)) for k, c in seen.items()}


def canon_list(values, canon):
    """Map each value to its canonical spelling, drop empties and case duplicates."""
    out, keys = [], set()
    for v in values:
        k = label_key(v)
        if k and k not in keys:
            keys.add(k)
            out.append(canon[k])
    return out


# Free-text fields whose values become question options on the site
LABEL_FIELDS = {"x": "connectsTo", "ind": "industries", "req": "requires"}


def build(with_vectors=True):
    registry = read_json(REGISTRY)
    needs = sorted((read_json(f) for f in NEEDS.glob("*.json")), key=lambda n: n["order"])
    need_index = {n["id"]: i for i, n in enumerate(needs)}
    pubs = load_publisher_files(registry)

    # Visible apps, in a stable order
    apps = []
    for f in sorted(APPS.glob("*.json")):
        a = read_json(f)
        m = effective_mapping(a)
        if m.get("hide") or a["source"].get("removedOn"):
            continue
        if a["source"].get("productType") == "SaaS" and not a["source"].get("mentionsBC"):
            continue
        apps.append((a, m))
    apps.sort(key=lambda am: (am[0]["source"]["name"].lower(), am[0]["appId"]))
    app_index = {a["appId"]: i for i, (a, _) in enumerate(apps)}
    canon = {f: canonical_labels(v for _, m in apps for v in m.get(f) or [])
             for f in LABEL_FIELDS.values()}

    out_apps, counts, warnings = [], [0] * len(needs), []
    for i, (a, m) in enumerate(apps):
        p = pubs.get(a["appId"], {})
        g = []
        for x in m.get("primary", []):
            if x["need"] in need_index:
                g.append(need_index[x["need"]])
            else:
                warnings.append(f"{a['appId']}: unknown need {x['need']}")
        g2 = [need_index[n] for n in m.get("secondary", []) if n in need_index]
        for gi in set(g):
            counts[gi] += 1
        pub_c = [c for c in p.get("countries", []) if c not in m.get("countries", [])]
        pub_l = [l for l in p.get("languages", []) if l not in m.get("languages", [])]
        row = {
            "i": i,
            "n": a["source"]["name"],
            "p": a["source"]["publisher"],
            "s": m.get("summary") or "",
            "t": m.get("shape") or "other",
            "g": g,
            "g2": g2,
            "c": merge(m.get("countries", []), pub_c),
            "l": merge(m.get("languages", []), pub_l),
            "x": works_with(a, canon_list(m.get("connectsTo") or [], canon["connectsTo"])),
            "ind": canon_list(m.get("industries") or [], canon["industries"]),
            "pr": m.get("pricing", "not_stated"),
            "req": canon_list(m.get("requires") or [], canon["requires"]),
            "k": m.get("claims", []),
            "u": a["source"]["url"],
        }
        if a["source"].get("allCountries") and not row["c"]:
            row["ac"] = 1  # listing says: all countries where BC is available
        if a["source"].get("productType") == "SaaS":
            row["h"] = 1  # hosted offer, not a BC extension
        # Publisher supplements, kept separate so the site can attribute them
        pub = {}
        if pub_c:
            pub["c"] = pub_c
        if pub_l:
            pub["l"] = pub_l
        ext = [app_index[x] for x in p.get("extends", []) if x in app_index]
        www = [app_index[x] for x in p.get("worksWellWith", []) if x in app_index]
        if ext:
            pub["ext"] = ext
        if www:
            pub["www"] = www
        if pub:
            row["pub"] = pub
        if p.get("showLogo") and p.get("logoUrl"):
            row["ic"] = p["logoUrl"]
        out_apps.append(row)

    groups = [{
        "id": n["id"], "label": n["label"], "desc": n["description"], "type": n["type"],
        "area": n["area"], "rel": n["relevance"], "q": n["questions"], "count": counts[i],
    } for i, n in enumerate(needs)]
    statements = [[i, s] for i, n in enumerate(needs) for s in n["statements"]]

    SITE_DATA.mkdir(parents=True, exist_ok=True)
    for name, obj in (("apps", out_apps), ("groups", groups), ("statements", statements)):
        (SITE_DATA / f"{name}.json").write_text(
            json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    for w in warnings[:10]:
        print("warn ", w)
    print(f"Built site/data: {len(out_apps)} apps, {len(groups)} needs, {len(statements)} statements, "
          f"{sum(1 for r in out_apps if 'pub' in r)} apps with publisher supplements, "
          f"{sum(1 for r in out_apps if 'ic' in r)} logos")

    if with_vectors:
        build_vectors(statements)
    else:
        drop_stale_vectors(statements)


def vectors_digest(statements):
    return hashlib.sha256(json.dumps([EMBED_MODEL, statements], ensure_ascii=False).encode()).hexdigest()


def drop_stale_vectors(statements):
    """Without vectors, never leave old ones behind: they would route to the wrong needs."""
    meta_path, vec_path = SITE_DATA / "vectors.meta.json", SITE_DATA / "vectors.bin"
    if vec_path.exists() and not (meta_path.exists() and read_json(meta_path).get("sha256") == vectors_digest(statements)):
        vec_path.unlink()
        meta_path.unlink(missing_ok=True)
        print("Statements changed: removed old vectors. The site uses keyword matching until you build with vectors.")


def build_vectors(statements):
    """Embed statements once; skip when the statement text is unchanged."""
    meta_path = SITE_DATA / "vectors.meta.json"
    vec_path = SITE_DATA / "vectors.bin"
    digest = vectors_digest(statements)
    if vec_path.exists() and meta_path.exists() and read_json(meta_path).get("sha256") == digest:
        print("Vectors unchanged, skipped")
        return
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("Vectors need sentence-transformers. Install it (pip install sentence-transformers), "
                 "or use --no-vectors (the site then falls back to keyword matching).")
    print(f"Embedding {len(statements)} statements (about a minute)...")
    model = SentenceTransformer(EMBED_MODEL)
    vectors = model.encode([s[1] for s in statements], batch_size=128,
                           normalize_embeddings=True, show_progress_bar=True).astype("<f4")
    vectors.tofile(vec_path)
    write_json(meta_path, {"model": EMBED_MODEL, "count": len(statements), "sha256": digest})
    print(f"Wrote vectors {vectors.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-vectors", action="store_true")
    build(not ap.parse_args().no_vectors)
