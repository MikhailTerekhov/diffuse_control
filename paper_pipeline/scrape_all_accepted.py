"""Stage 2: Scrape ALL accepted papers (oral, spotlight, poster) for utils.VENUES.

Adds new papers to metadata.json without overwriting existing ones.
Preserves decision type (oral, spotlight, poster) in metadata.

Usage:
  python scrape_all_accepted.py
"""

import time
import openreview
from utils import VENUES, paper_id, load_metadata, save_metadata

ACCEPTED_DECISIONS = {
    "oral": "oral",
    "spotlight": "spotlight",
    "spotlightposter": "spotlight",
    "poster": "poster",
    "conditionalposter": "poster",
    "conditionaloral": "oral",
}


def classify_decision(venue_str: str) -> str | None:
    """Extract decision type from venue string. Returns None if not accepted."""
    lower = venue_str.lower()
    for key, decision in ACCEPTED_DECISIONS.items():
        if key in lower:
            return decision
    return None


def scrape_venue(client, venue_id: str, conf: str, year: int) -> list[dict]:
    print(f"\nQuerying {conf} {year} ({venue_id})...")

    submissions = client.get_all_notes(
        invitation=f"{venue_id}/-/Submission",
        details="directReplies",
    )
    print(f"  Found {len(submissions)} total submissions")

    papers = []
    by_decision = {}

    for note in submissions:
        content = note.content or {}

        venue_field = content.get("venue", {})
        venue_str = venue_field.get("value", "") if isinstance(venue_field, dict) else str(venue_field)

        decision = classify_decision(venue_str)
        if not decision:
            # Check venueid as fallback
            venueid_field = content.get("venueid", {})
            venueid_str = venueid_field.get("value", "") if isinstance(venueid_field, dict) else str(venueid_field)
            decision = classify_decision(venueid_str)

        if not decision:
            continue

        title_field = content.get("title", {})
        title = title_field.get("value", "") if isinstance(title_field, dict) else str(title_field)

        authors_field = content.get("authors", {})
        authors = authors_field.get("value", []) if isinstance(authors_field, dict) else []

        pid = paper_id(conf, year, note.id)
        papers.append({
            "paper_id": pid,
            "openreview_id": note.id,
            "conference": conf,
            "year": year,
            "decision": decision,
            "title": title,
            "authors": authors,
        })
        by_decision[decision] = by_decision.get(decision, 0) + 1

    print(f"  Accepted: {len(papers)} total")
    for d, c in sorted(by_decision.items()):
        print(f"    {d}: {c}")
    return papers


def run():
    client = openreview.api.OpenReviewClient(baseurl="https://api2.openreview.net")
    metadata = load_metadata()

    new_count = 0
    updated_count = 0

    for venue_id, conf, year in VENUES:
        papers = scrape_venue(client, venue_id, conf, year)
        for p in papers:
            pid = p["paper_id"]
            if pid in metadata["papers"]:
                # Update decision if it changed (e.g. was "oral", now has more detail)
                if metadata["papers"][pid].get("decision") != p["decision"]:
                    metadata["papers"][pid]["decision"] = p["decision"]
                    updated_count += 1
            else:
                metadata["papers"][pid] = {
                    "openreview_id": p["openreview_id"],
                    "conference": p["conference"],
                    "year": p["year"],
                    "decision": p["decision"],
                    "title": p["title"],
                    "authors": p["authors"],
                    "arxiv_id": None,
                    "arxiv_url": None,
                    "arxiv_first_submitted": None,
                    "pdf_downloaded": False,
                    "has_supplementary": False,
                    "pdf_pages": None,
                    "text_extracted": False,
                    "figures_extracted": False,
                    "figure_count": None,
                    "paper_classified": False,
                    "rq_extracted": False,
                    "error": None,
                }
                new_count += 1
        save_metadata(metadata)
        time.sleep(1)

    print(f"\nAdded {new_count} new papers, updated {updated_count} decisions")

    total = len(metadata["papers"])
    by_venue = {}
    for p in metadata["papers"].values():
        key = f"{p['conference']} {p['year']}"
        by_venue.setdefault(key, {"oral": 0, "spotlight": 0, "poster": 0, "total": 0})
        by_venue[key][p.get("decision", "poster")] += 1
        by_venue[key]["total"] += 1

    print(f"\nTotal papers: {total}")
    for k in sorted(by_venue):
        v = by_venue[k]
        print(f"  {k}: {v['total']} (oral={v['oral']}, spotlight={v['spotlight']}, poster={v['poster']})")


if __name__ == "__main__":
    run()
