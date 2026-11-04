"""Stage 1: Fetch spotlight/oral paper metadata from OpenReview."""

import time
import openreview
from tqdm import tqdm
from utils import VENUES, paper_id, load_metadata, save_metadata


def get_client() -> openreview.api.OpenReviewClient:
    return openreview.api.OpenReviewClient(baseurl="https://api2.openreview.net")


def decision_matches(decision: str) -> bool:
    """Check if a decision string indicates spotlight or oral."""
    d = decision.lower()
    return any(kw in d for kw in ["spotlight", "oral"])


def scrape_venue(client: openreview.api.OpenReviewClient,
                 venue_id: str, conf: str, year: int) -> list[dict]:
    """Fetch accepted spotlight/oral papers from a venue."""
    papers = []
    print(f"\nQuerying {conf} {year} ({venue_id})...")

    # Get all accepted submissions
    try:
        submissions = client.get_all_notes(
            invitation=f"{venue_id}/-/Submission",
            details="directReplies",
        )
    except Exception as e:
        print(f"  Error fetching submissions: {e}")
        # Try alternative invitation format
        try:
            submissions = client.get_all_notes(content={"venueid": venue_id})
        except Exception as e2:
            print(f"  Fallback also failed: {e2}")
            return papers

    print(f"  Found {len(submissions)} total submissions")

    for note in submissions:
        content = note.content or {}

        # Check decision — stored in venue field or in direct replies
        venue_field = content.get("venue", {})
        venue_str = venue_field.get("value", "") if isinstance(venue_field, dict) else str(venue_field)

        # Also check venueid
        venueid_field = content.get("venueid", {})
        venueid_str = venueid_field.get("value", "") if isinstance(venueid_field, dict) else str(venueid_field)

        # Determine decision from venue string
        decision = ""
        for s in [venue_str, venueid_str]:
            if decision_matches(s):
                decision = "spotlight" if "spotlight" in s.lower() else "oral"
                break

        if not decision:
            # Check direct replies for decision notes
            for reply in (note.details.get("directReplies", []) if note.details else []):
                reply_content = reply.get("content", {})
                dec_field = reply_content.get("decision", {})
                dec_str = dec_field.get("value", "") if isinstance(dec_field, dict) else str(dec_field)
                if decision_matches(dec_str):
                    decision = "spotlight" if "spotlight" in dec_str.lower() else "oral"
                    break

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

    print(f"  Found {len(papers)} spotlight/oral papers")
    return papers


def run():
    client = get_client()
    metadata = load_metadata()

    for venue_id, conf, year in VENUES:
        papers = scrape_venue(client, venue_id, conf, year)
        for p in papers:
            pid = p["paper_id"]
            if pid not in metadata["papers"]:
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
        save_metadata(metadata)
        time.sleep(1)  # courtesy delay between venues

    total = len(metadata["papers"])
    by_venue = {}
    for p in metadata["papers"].values():
        key = f"{p['conference']} {p['year']}"
        by_venue[key] = by_venue.get(key, 0) + 1

    print(f"\nTotal papers: {total}")
    for k, v in sorted(by_venue.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    run()
