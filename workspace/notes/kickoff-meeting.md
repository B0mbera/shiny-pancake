# Kickoff meeting: summarization hub

Date: 2026-10-27. Attendees: Anna (product), Bence (engineering), Dora (security), Gabor (operations).

## Context

Reviewers spend about six hours a week reading status reports and incident write-ups. The goal of
the summarization hub is to cut that to one hour while keeping every decision and action item.
Anna explained that reviewers care most about decisions, owners and deadlines, and less about
background sections.

## Decisions

- We will use the ensemble strategy with key points, action items and risks as the default profiles.
- Summary workers run on the small model with low effort; the editor runs on the main model.
- Raw documents stay inside the workspace sandbox and are never sent to third-party tools.

## Action items

- Bence will ship the MCP bridge and the job queue by Friday, November 7.
- Dora must review the write_file tool and set the approval policy to deny in production.
- Gabor needs to configure cost alerts at 5 dollars per day before the pilot starts.
- Anna will collect twenty real reports for the evaluation suite by November 10.
- The team should schedule a review of reviewer ratings two weeks after the pilot launch.

## Risks and open questions

- Prompt injection hidden in incident reports is a real risk, because reports quote customer emails.
- Latency could become a problem if the provider rate limits us during the Monday morning peak.
- It is still an open question whether the refine strategy reads better for long incident timelines.
- We assume reviewers will accept summaries without links back to the source paragraphs.
